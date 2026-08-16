NVCC       = nvcc
NVCC_FLAGS = -std=c++14 -O2
NPP_LIBS   = -lnppif -lnppc

TARGET = image_processor
SRC    = src/main.cu

.PHONY: all build clean data run

all: $(TARGET)

build: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NPP_LIBS)

data:
	python3 scripts/generate_test_data.py data 100 128

run: $(TARGET) data
	mkdir -p output
	./$(TARGET) --input data --output output --filter edge

clean:
	rm -f $(TARGET)
	rm -rf output data
