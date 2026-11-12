# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# Build:   CUDA_VER=12.6 make
# Install: CUDA_VER=12.6 make install
# Optional NMSX_LIB=/path/libnmsx.so links the EfficientNMSX_TRT plugin library, so nvinfer loads it with the parser.

ifeq ($(CUDA_VER),)
  $(error CUDA_VER is not set, e.g. CUDA_VER=12.6 make)
endif

DS_DIR ?= /opt/nvidia/deepstream/deepstream
CC := g++
CFLAGS := -Wall -std=c++11 -shared -fPIC
CFLAGS += -I$(DS_DIR)/sources/includes -I/usr/local/cuda-$(CUDA_VER)/include
LIBS :=
ifneq ($(NMSX_LIB),)
  LIBS += -Wl,--no-as-needed $(abspath $(NMSX_LIB)) -Wl,-rpath,$(dir $(abspath $(NMSX_LIB)))
endif

SRCFILES := nvdsinfer_yolo_attr.cpp
TARGET_LIB := libnvds_infer_yolo_attr.so

all: $(TARGET_LIB)

$(TARGET_LIB): $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

install: $(TARGET_LIB)
	cp $(TARGET_LIB) $(DS_DIR)/lib/

clean:
	rm -rf $(TARGET_LIB)
