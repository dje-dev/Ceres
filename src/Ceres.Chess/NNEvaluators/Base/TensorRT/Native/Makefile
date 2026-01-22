CXX = g++
CXXFLAGS = -std=c++17 -fPIC -O2 -Wall -fvisibility=hidden
CUDA_PATH = /usr/local/cuda-13.0
INCLUDES = -I/usr/include/aarch64-linux-gnu -I$(CUDA_PATH)/targets/sbsa-linux/include
LDFLAGS = -shared -L/usr/lib/aarch64-linux-gnu -L$(CUDA_PATH)/targets/sbsa-linux/lib
LIBS = -lnvinfer -lnvonnxparser -lcudart

TARGET = libTensorRTWrapper.so
SOURCES = TensorRTWrapper.cpp
OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

clean:
	rm -f $(OBJECTS) $(TARGET)

install: $(TARGET)
	cp $(TARGET) ..

.PHONY: all clean install
