// CUDA Batch Image Processing Pipeline
// Applies grayscale conversion, Gaussian blur, and Sobel edge detection
// to batches of PPM images using custom CUDA kernels and NVIDIA NPP.

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <npp.h>
#include <nppi_filtering_functions.h>

// ---------------------------------------------------------------------------
// Error-checking macros
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t err = (call);                                               \
    if (err != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
              cudaGetErrorString(err));                                      \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
  } while (0)

#define NPP_CHECK(call)                                                     \
  do {                                                                      \
    NppStatus status = (call);                                              \
    if (status != NPP_SUCCESS) {                                            \
      fprintf(stderr, "NPP error at %s:%d: status %d\n", __FILE__,         \
              __LINE__, static_cast<int>(status));                          \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
  } while (0)

// ---------------------------------------------------------------------------
// PPM image I/O
// ---------------------------------------------------------------------------

struct Image {
  int width;
  int height;
  int channels;
  unsigned char* data;
};

static void SkipPpmWhitespaceAndComments(FILE* fp) {
  int c;
  while ((c = fgetc(fp)) != EOF) {
    if (c == '#') {
      while ((c = fgetc(fp)) != '\n' && c != EOF) {}
    } else if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
      ungetc(c, fp);
      return;
    }
  }
}

static Image ReadPpm(const char* filename) {
  FILE* fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Error: cannot open %s\n", filename);
    exit(EXIT_FAILURE);
  }

  char magic[3] = {0};
  if (fread(magic, 1, 2, fp) != 2 || strcmp(magic, "P6") != 0) {
    fprintf(stderr, "Error: %s is not a valid P6 PPM file\n", filename);
    fclose(fp);
    exit(EXIT_FAILURE);
  }

  int width = 0, height = 0, max_val = 0;
  SkipPpmWhitespaceAndComments(fp);
  if (fscanf(fp, "%d", &width) != 1) { fclose(fp); exit(EXIT_FAILURE); }
  SkipPpmWhitespaceAndComments(fp);
  if (fscanf(fp, "%d", &height) != 1) { fclose(fp); exit(EXIT_FAILURE); }
  SkipPpmWhitespaceAndComments(fp);
  if (fscanf(fp, "%d", &max_val) != 1) { fclose(fp); exit(EXIT_FAILURE); }
  fgetc(fp);  // consume single whitespace before binary data

  if (width <= 0 || height <= 0) {
    fprintf(stderr, "Error: invalid dimensions %dx%d in %s\n",
            width, height, filename);
    fclose(fp);
    exit(EXIT_FAILURE);
  }

  Image img;
  img.width = width;
  img.height = height;
  img.channels = 3;
  size_t data_size = static_cast<size_t>(width) * height * 3;
  img.data = static_cast<unsigned char*>(malloc(data_size));
  if (fread(img.data, 1, data_size, fp) != data_size) {
    fprintf(stderr, "Warning: incomplete read for %s\n", filename);
  }
  fclose(fp);
  return img;
}

static void WritePpm(const char* filename, const unsigned char* data,
                     int width, int height, int channels) {
  FILE* fp = fopen(filename, "wb");
  if (!fp) {
    fprintf(stderr, "Error: cannot write to %s\n", filename);
    return;
  }
  fprintf(fp, "P6\n%d %d\n255\n", width, height);
  if (channels == 1) {
    for (int i = 0; i < width * height; ++i) {
      unsigned char v = data[i];
      fputc(v, fp);
      fputc(v, fp);
      fputc(v, fp);
    }
  } else {
    fwrite(data, 1, width * height * 3, fp);
  }
  fclose(fp);
}

static void FreeImage(Image* img) {
  free(img->data);
  img->data = nullptr;
}

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// RGB to grayscale using the luminosity method.
__global__ void GrayscaleKernel(const unsigned char* __restrict__ rgb,
                                unsigned char* __restrict__ gray,
                                int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    int rgb_idx = (y * width + x) * 3;
    int gray_idx = y * width + x;
    gray[gray_idx] = static_cast<unsigned char>(
        0.299f * rgb[rgb_idx] +
        0.587f * rgb[rgb_idx + 1] +
        0.114f * rgb[rgb_idx + 2]);
  }
}

// Sobel edge detection on a single-channel image.
__global__ void SobelKernel(const unsigned char* __restrict__ input,
                            unsigned char* __restrict__ output,
                            int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
    float gx =
        -1.0f * input[(y - 1) * width + (x - 1)] +
         1.0f * input[(y - 1) * width + (x + 1)] +
        -2.0f * input[y       * width + (x - 1)] +
         2.0f * input[y       * width + (x + 1)] +
        -1.0f * input[(y + 1) * width + (x - 1)] +
         1.0f * input[(y + 1) * width + (x + 1)];

    float gy =
        -1.0f * input[(y - 1) * width + (x - 1)] +
        -2.0f * input[(y - 1) * width + x]        +
        -1.0f * input[(y - 1) * width + (x + 1)] +
         1.0f * input[(y + 1) * width + (x - 1)] +
         2.0f * input[(y + 1) * width + x]        +
         1.0f * input[(y + 1) * width + (x + 1)];

    float mag = sqrtf(gx * gx + gy * gy);
    output[y * width + x] = static_cast<unsigned char>(fminf(mag, 255.0f));
  } else if (x < width && y < height) {
    output[y * width + x] = 0;
  }
}

// ---------------------------------------------------------------------------
// Image processing pipeline
// ---------------------------------------------------------------------------

enum FilterMode {
  kGrayscale = 1,
  kBlur      = 2,
  kEdge      = 4,
  kAll       = 7
};

static void ProcessImage(const Image& img, const char* output_path,
                         int filter_mode) {
  int w = img.width;
  int h = img.height;
  size_t rgb_size  = static_cast<size_t>(w) * h * 3;
  size_t gray_size = static_cast<size_t>(w) * h;

  unsigned char* d_rgb     = nullptr;
  unsigned char* d_gray    = nullptr;
  unsigned char* d_blurred = nullptr;
  unsigned char* d_edges   = nullptr;

  CUDA_CHECK(cudaMalloc(&d_rgb, rgb_size));
  CUDA_CHECK(cudaMalloc(&d_gray, gray_size));
  CUDA_CHECK(cudaMemcpy(d_rgb, img.data, rgb_size, cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid((w + 15) / 16, (h + 15) / 16);

  // Stage 1: grayscale conversion (custom kernel)
  GrayscaleKernel<<<grid, block>>>(d_rgb, d_gray, w, h);
  CUDA_CHECK(cudaGetLastError());

  unsigned char* d_result = d_gray;

  // Stage 2: Gaussian blur (NPP library)
  if (filter_mode & kBlur) {
    CUDA_CHECK(cudaMalloc(&d_blurred, gray_size));
    NppiSize roi = {w, h};
    NPP_CHECK(nppiFilterGauss_8u_C1R(
        d_gray, w, d_blurred, w, roi, NPP_MASK_SIZE_5_X_5));
    d_result = d_blurred;
  }

  // Stage 3: Sobel edge detection (custom kernel)
  if (filter_mode & kEdge) {
    CUDA_CHECK(cudaMalloc(&d_edges, gray_size));
    unsigned char* sobel_input = d_blurred ? d_blurred : d_gray;
    SobelKernel<<<grid, block>>>(sobel_input, d_edges, w, h);
    CUDA_CHECK(cudaGetLastError());
    d_result = d_edges;
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  unsigned char* h_output = static_cast<unsigned char*>(malloc(gray_size));
  CUDA_CHECK(cudaMemcpy(h_output, d_result, gray_size,
                         cudaMemcpyDeviceToHost));
  WritePpm(output_path, h_output, w, h, 1);

  free(h_output);
  CUDA_CHECK(cudaFree(d_rgb));
  CUDA_CHECK(cudaFree(d_gray));
  if (d_blurred) CUDA_CHECK(cudaFree(d_blurred));
  if (d_edges)   CUDA_CHECK(cudaFree(d_edges));
}

// ---------------------------------------------------------------------------
// File utilities
// ---------------------------------------------------------------------------

static bool HasImageExtension(const char* filename) {
  const char* ext = strrchr(filename, '.');
  if (!ext) return false;
  return strcmp(ext, ".ppm") == 0 || strcmp(ext, ".PPM") == 0;
}

static std::vector<std::string> ListImageFiles(const char* dir_path) {
  std::vector<std::string> files;
  DIR* dir = opendir(dir_path);
  if (!dir) {
    fprintf(stderr, "Error: cannot open directory %s\n", dir_path);
    return files;
  }
  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    if (HasImageExtension(entry->d_name)) {
      files.push_back(std::string(dir_path) + "/" + entry->d_name);
    }
  }
  closedir(dir);
  std::sort(files.begin(), files.end());
  return files;
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

static void PrintUsage(const char* program) {
  printf("Usage: %s --input <dir> --output <dir> [--filter <mode>]\n\n",
         program);
  printf("Filter modes:\n");
  printf("  grayscale  Convert to grayscale only\n");
  printf("  blur       Grayscale + Gaussian blur\n");
  printf("  edge       Full pipeline: grayscale + blur + Sobel (default)\n");
}

static int ParseFilterMode(const char* mode_str) {
  if (strcmp(mode_str, "grayscale") == 0) return kGrayscale;
  if (strcmp(mode_str, "blur") == 0)      return kGrayscale | kBlur;
  if (strcmp(mode_str, "edge") == 0)      return kAll;
  if (strcmp(mode_str, "all") == 0)       return kAll;
  fprintf(stderr, "Unknown filter mode: %s\n", mode_str);
  return -1;
}

static const char* FilterModeLabel(int mode) {
  if (mode == kGrayscale)            return "grayscale";
  if (mode == (kGrayscale | kBlur))  return "grayscale -> blur";
  if (mode == kAll)                  return "grayscale -> blur -> edge";
  return "custom";
}

int main(int argc, char* argv[]) {
  const char* input_dir  = nullptr;
  const char* output_dir = nullptr;
  int filter_mode = kAll;

  for (int i = 1; i < argc; ++i) {
    if ((strcmp(argv[i], "--input") == 0 || strcmp(argv[i], "-i") == 0) &&
        i + 1 < argc) {
      input_dir = argv[++i];
    } else if ((strcmp(argv[i], "--output") == 0 ||
                strcmp(argv[i], "-o") == 0) && i + 1 < argc) {
      output_dir = argv[++i];
    } else if ((strcmp(argv[i], "--filter") == 0 ||
                strcmp(argv[i], "-f") == 0) && i + 1 < argc) {
      filter_mode = ParseFilterMode(argv[++i]);
      if (filter_mode < 0) return EXIT_FAILURE;
    } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      PrintUsage(argv[0]);
      return 0;
    }
  }

  if (!input_dir || !output_dir) {
    PrintUsage(argv[0]);
    return EXIT_FAILURE;
  }

  // Print GPU information
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
  printf("Pipeline: %s\n\n", FilterModeLabel(filter_mode));

  // Discover input images
  std::vector<std::string> files = ListImageFiles(input_dir);
  if (files.empty()) {
    fprintf(stderr, "No PPM images found in %s\n", input_dir);
    return EXIT_FAILURE;
  }
  printf("Found %zu images in %s\n\n", files.size(), input_dir);

  // Create output directory (ignore error if it already exists)
  mkdir(output_dir, 0755);

  // Process batch with GPU timing
  cudaEvent_t t_start, t_stop;
  CUDA_CHECK(cudaEventCreate(&t_start));
  CUDA_CHECK(cudaEventCreate(&t_stop));
  CUDA_CHECK(cudaEventRecord(t_start));

  for (size_t i = 0; i < files.size(); ++i) {
    const char* base = strrchr(files[i].c_str(), '/');
    base = base ? base + 1 : files[i].c_str();

    std::string out_path = std::string(output_dir) + "/" + base;

    printf("[%3zu/%zu] %s\n", i + 1, files.size(), base);

    Image img = ReadPpm(files[i].c_str());
    ProcessImage(img, out_path.c_str(), filter_mode);
    FreeImage(&img);
  }

  CUDA_CHECK(cudaEventRecord(t_stop));
  CUDA_CHECK(cudaEventSynchronize(t_stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, t_start, t_stop));

  printf("\n=== Summary ===\n");
  printf("Images processed : %zu\n", files.size());
  printf("Total GPU time   : %.2f ms\n", elapsed_ms);
  printf("Avg per image    : %.2f ms\n", elapsed_ms / files.size());
  printf("Pipeline         : %s\n", FilterModeLabel(filter_mode));
  printf("Output directory : %s/\n", output_dir);

  CUDA_CHECK(cudaEventDestroy(t_start));
  CUDA_CHECK(cudaEventDestroy(t_stop));
  CUDA_CHECK(cudaDeviceReset());
  return 0;
}
