OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui
OPENCV_LIBPATH=/usr/lib64
OPENCV_INCLUDEPATH=/usr/include
CUDA_INCLUDEPATH=/usr/local/cuda-5.5/include
CUDA_LIBPATH=/usr/local/cuda-5.5/lib64
CUDA_LIBS=-lcudart

NVCC_OPTS=-O3 -arch=sm_21 -Xcompiler -Wall -Xcompiler -Wextra

GCC_OPTS=-O3 -Wall -Wextra

rgb_to_grey:
	nvcc -o rgb_to_grey $(NVCC_OPTS) -I$(OPENCV_INCLUDEPATH) -I$(CUDA_INCLUDEPATH) -L$(CUDA_LIBPATH) $(CUDA_LIBS) -L$(OPENCV_LIBPATH) $(OPENCV_LIBS) rgb_to_grey.cu

clean:
	rm -rf rgb_to_grey output.jpg