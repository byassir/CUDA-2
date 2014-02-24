/*
  Author: Faiz Ilahi Kothari
  email: faiz.off93@gmail.com
  Date modified: 24/02/2014
  Description: Converts any image to greyscale. Intensity computed according to weights.
  			   Optimal Blocksize chosen. Converts a 9372X9372 (14.0 MB) image to greyscale in 54 millisec.
  			   Per block max no. of threads are 1024. In my case 16X16 block size is optimal.
  			   4X faster than the internal OpenCV conversion.
  Credits: Udacity

  TODO:
 */
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
#include <time.h>

#define BLOCK_X 16 //Optimal Block size in X for this code
#define BLOCK_Y 16 //Optimal Block size in Y for this code
#define WEIGHT_RED 0.299f 
#define WEIGHT_GREEN 0.587f   //Eyes most sensitive to green and least to blue. Alpha channel is ignored
#define WEIGHT_BLUE 0.114f

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line){
	if(err != cudaSuccess){
    	std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    	std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    	exit(1);
    }
}

struct GpuTimer //Timer to calculate the time taken by the kernel to execute
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

//KERNEL CODE

__global__ void kernel_rgb_to_grey(uchar4 *imgRGBA, unsigned char *imgGrey, int numCols, int numRows) 
{
	int pixel_x = ((blockDim.x * blockIdx.x) + threadIdx.x);
	int pixel_y = ((blockDim.y * blockIdx.y) + threadIdx.y);
	int pos = (pixel_y * numCols) + pixel_x;
	if((pixel_x < numCols) && (pixel_y < numRows)){
		imgGrey[pos] = (WEIGHT_RED * imgRGBA[pos].x + WEIGHT_GREEN * imgRGBA[pos].y + WEIGHT_BLUE * imgRGBA[pos].z );
	}
}

int main(int argc, char **argv)
{
	cv::Mat imageRGBA;
	cv::Mat imageGrey;

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
	int g_x = ((BLOCK_X - (imageRGBA.cols % BLOCK_X)) + imageRGBA.cols) / BLOCK_X;
	int g_y = ((BLOCK_Y - (imageRGBA.rows % BLOCK_Y)) + imageRGBA.rows) / BLOCK_Y;

	int b_x = BLOCK_X;
	int b_y = BLOCK_Y; //Max Threads per block is 1024, so 32x32

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
	
	/*
	//Compare with OpenCV inbuilt conversion
	clock_t t = clock();
	cv::cvtColor(image, imageGrey, CV_BGR2GRAY);
	t = clock() - t;
	std::cout << "Time for OpenCV to convert the image: " << ((float)t)/CLOCKS_PER_SEC << std::endl;
	cv::imwrite((std::string("output_ocv.jpg")).c_str(), imageGrey);
	//Comment when used.
	*/

	//Free all the memory
	cudaFree(d_imgRGBA);
	cudaFree(d_imgGrey);

	return 0;
}