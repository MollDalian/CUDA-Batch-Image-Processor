# CUDA Batch Image Processing Pipeline

A GPU-accelerated image processing pipeline that applies a chain of filters
to batches of PPM images. Demonstrates both custom CUDA kernels and the
NVIDIA Performance Primitives (NPP) library for high-throughput image
processing.

## Processing Pipeline

Each image passes through up to three stages:

1. **Grayscale conversion** (custom CUDA kernel) -- Converts RGB to single-channel
   grayscale using the luminosity method: `0.299*R + 0.587*G + 0.114*B`.
2. **Gaussian blur** (NVIDIA NPP) -- Applies a 5x5 Gaussian smoothing filter
   via `nppiFilterGauss_8u_C1R`. Pre-blurring reduces noise before edge detection.
3. **Sobel edge detection** (custom CUDA kernel) -- Computes horizontal and vertical
   gradients with 3x3 Sobel operators and outputs the gradient magnitude.

## GPU Algorithms

### Grayscale Kernel

Uses a 2D thread grid (16x16 blocks) mapped to image coordinates. Each thread
processes one pixel, computing the weighted sum of the three RGB channels. The
weights (0.299, 0.587, 0.114) match the ITU-R BT.601 standard used by most
image-editing software.

### NPP Gaussian Blur

Calls `nppiFilterGauss_8u_C1R` from NVIDIA's NPP library with a 5x5 mask.
NPP's implementation is heavily optimized for GPU memory access patterns, making
it faster than a naive custom kernel for standard convolution filters.

### Sobel Edge Detection Kernel

Each thread computes horizontal (Gx) and vertical (Gy) gradients using the
standard 3x3 Sobel operators:

```
Gx:  -1  0  +1      Gy:  -1  -2  -1
     -2  0  +2            0   0   0
     -1  0  +1           +1  +2  +1
```

The output pixel value is the gradient magnitude: `sqrt(Gx^2 + Gy^2)`, clamped
to [0, 255]. Border pixels are set to 0.

## Prerequisites

- CUDA Toolkit 10.0+ with NPP libraries
- Python 3 (for test data generation, no external packages needed)
- Linux with a CUDA-capable GPU

## Quick Start

```bash
chmod +x run.sh
./run.sh
```

This builds the project, generates 100 test images, and runs the full pipeline.

## Manual Build and Run

```bash
# Build
make

# Generate test images (100 images, 128x128 pixels)
python3 scripts/generate_test_data.py data 100 128

# Run the processor
./image_processor --input data --output output --filter edge
```

Or use the Makefile shortcut:

```bash
make run
```

## CLI Arguments

| Flag | Short | Description |
|------|-------|-------------|
| `--input <dir>` | `-i` | Input directory containing PPM images |
| `--output <dir>` | `-o` | Output directory for processed images |
| `--filter <mode>` | `-f` | Filter pipeline (see below) |
| `--help` | `-h` | Show usage |

**Filter modes:**

- `grayscale` -- Grayscale conversion only
- `blur` -- Grayscale + Gaussian blur
- `edge` -- Full pipeline: grayscale + blur + Sobel edge detection (default)

## Test Data

The included Python script generates PPM images with no external dependencies:

```bash
python3 scripts/generate_test_data.py [output_dir] [count] [size]
```

- Defaults: 100 images, 128x128 pixels, written to `data/`
- Patterns: gradients, checkerboards, concentric circles, stripes, sine waves, and random noise

## Project Structure

```
.
├── src/
│   └── main.cu                  # CUDA source (kernels + pipeline + CLI)
├── scripts/
│   └── generate_test_data.py    # Test image generator (pure Python)
├── samples/
│   └── execution_log.txt        # Full build + run log (proof of execution)
├── Makefile                     # Build system
├── run.sh                       # One-command build-and-run script
└── README.md
```

After running, two directories are created:

```
├── data/      # Generated input images (PPM)
└── output/    # Processed output images (PPM)
```

## Sample Output

```
GPU: Tesla T4 (Compute 7.5)
Pipeline: grayscale -> blur -> edge

Found 100 images in data

[  1/100] gradient_0000.ppm
[  2/100] checker_0001.ppm
...
[100/100] noise_0099.ppm

=== Summary ===
Images processed : 100
Total GPU time   : 187.34 ms
Avg per image    : 1.87 ms
Pipeline         : grayscale -> blur -> edge
Output directory : output/
```

## Lessons Learned

- **NPP vs. custom kernels:** NPP provides heavily optimized implementations for
  standard operations like Gaussian blur, but custom kernels are necessary for
  non-standard algorithms or when fine-grained control over thread mapping is needed.
  Combining both in one pipeline is a practical pattern for real workloads.

- **Memory management trade-offs:** Allocating and freeing GPU memory per image is
  simple and correct but adds overhead from repeated `cudaMalloc`/`cudaFree` calls.
  A production pipeline would pre-allocate buffers for the maximum expected image
  size and reuse them across the batch.

- **2D thread grids for images:** Using 16x16 thread blocks for image processing
  maps naturally to the 2D pixel layout and achieves good occupancy on most GPUs.
  The grid dimensions are computed from the image size, so arbitrary resolutions
  are supported without wasted threads on aligned boundaries.

- **Blur before edge detection:** Applying Gaussian smoothing before Sobel edge
  detection significantly reduces false edges from sensor noise -- a standard
  technique in computer vision pipelines (used in the Canny edge detector, for
  example).
