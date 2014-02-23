#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <string>
#include <iomanip>

//#define DIMCOLS 557
//#define DIMROWS 313

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line){
	if(err != cudaSuccess){
    	std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    	std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    	exit(1);
    }
}

struct GpuTimer
{
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer()
  {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer()
  {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start()
  {
    cudaEventRecord(start, 0);
  }

  void Stop()
  {
    cudaEventRecord(stop, 0);
  }

  float Elapsed()
  {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

cv::Mat imageRGBA;
cv::Mat imageGrey;

__global__ void kernel_rgb_to_grey(uchar4 *imgRGBA, unsigned char *imgGrey, int numCols, int numRows)
{
	int pixel_x = (blockDim.x * blockIdx.x) + threadIdx.x;
	int pixel_y = (blockDim.y * blockIdx.y) + threadIdx.y;

	if((pixel_x < numCols) && (pixel_y < numRows)){
		imgGrey[(pixel_y * numCols) + pixel_x] = (0.299f * imgRGBA[(pixel_y * numCols) + pixel_x].x + 0.587f * imgRGBA[(pixel_y * numCols) + pixel_x].y + 0.114f * imgRGBA[(pixel_y * numCols) + pixel_x].z );
	}
}

int main(int argc, char **argv)
{
	if (argc != 3){
		std::cout << "Wrong no. of arguments!\n" << std::endl;
		exit(1);
	}

	checkCudaErrors(cudaFree(0)); //Making sure Context initializes OK

	std::string input_filename = std::string(argv[1]);
	std::string output_filename = std::string(argv[2]);

	cv::Mat image = cv::imread(input_filename.c_str(),CV_LOAD_IMAGE_COLOR);
	if(image.empty()){
		std::cout << "Improper image\n" << std::endl;
		exit(1);
	}

	cv::cvtColor(image, imageRGBA, CV_BGR2RGBA); //Create imageRGBA
	imageGrey.create(image.rows, image.cols, CV_8UC1); //Create ImageGrey

	const size_t numPixels = imageRGBA.rows * imageRGBA.cols; //Total no. of Pixels

	//Images on Host
	uchar4 *h_imgRGBA = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
	unsigned char *h_imgGrey = imageGrey.ptr<unsigned char>(0);

	//Images on Device
	uchar4 *d_imgRGBA = NULL; //d_imgRGBA is a double pointer because we need a pointer to a pointer to an image.
	unsigned char *d_imgGrey = NULL;  //Image is refered through image = pointer to uchar4. cudaMalloc needs a pointer to this pointer.

	//Allocate memory on GPU for both input and output
	checkCudaErrors(cudaMalloc(&d_imgRGBA, numPixels * sizeof(uchar4)));
	checkCudaErrors(cudaMalloc(&d_imgGrey, numPixels * sizeof(unsigned char)));

	//Memset memory on GPU for d_imgGrey so as to remove stale data
	checkCudaErrors(cudaMemset(d_imgGrey, 0, numPixels * sizeof(unsigned char)));

	//Copy memory from host to CUDA
	checkCudaErrors(cudaMemcpy(d_imgRGBA, h_imgRGBA, numPixels * sizeof(uchar4), cudaMemcpyHostToDevice));

	//Compute the dimensions of grid and blocks
	int g_x = ((32 - (imageRGBA.cols % 32)) + imageRGBA.cols) / 32;
	int g_y = ((32 - (imageRGBA.rows % 32)) + imageRGBA.rows) / 32;

	int b_x = 32;
	int b_y = 32; //Max Threads per block is 1024, so 32x32

	//Ready to launch the kernel
	GpuTimer timer;
	timer.Start();
	kernel_rgb_to_grey <<< dim3(g_x,g_y,1),dim3(b_x,b_y,1) >>> (d_imgRGBA, d_imgGrey, imageRGBA.cols, imageRGBA.rows);
	timer.Stop();
	std::cout << "Your code ran in: " << timer.Elapsed() << " msecs." << std::endl;
	//Copy Memory from GPU to Host
	checkCudaErrors(cudaMemcpy(h_imgGrey, d_imgGrey, numPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	//Create the output Image
	cv::Mat output(imageRGBA.rows, imageRGBA.cols, CV_8UC1, (void *)h_imgGrey);
	//output the image
	cv::imwrite(output_filename.c_str(), output);

	//Free all the memory
	cudaFree(d_imgRGBA);
	cudaFree(d_imgGrey);

	return 0;
}