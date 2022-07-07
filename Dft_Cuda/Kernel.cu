//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>
//#include <thrust/complex.h>

#include "Kernel.h"

using namespace thrust;

__device__ const float M_PI = 3.1415926535897932384626433832795f;

__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__device__ inline int index(int x, int y, int w) {
	return (y * w + x);
}


__device__ complex<float> sumCol(const float *im, const int x, const int w, const int h, const int u, const int v)
{
	float dx = float(x);
	float dw = float(w);
	float dh = float(h);
	float du = float(u);
	float dv = float(v);

	complex<float> fxy;
	complex<float> e;
	complex<float> expe;
	complex<float> mult;
	complex<float> currentSum;


	currentSum = complex<float>(0.0f, 0.0f);
	for (int y = 0; y < h; y++) {
		float dy = float(y);

		fxy = complex<float>(im[index(x, y, w)], 0);
		e = complex<float>(0, -2 * M_PI * ((dx * du) / dw + (dy * dv) / dh));
		expe = exp(e);
		mult = fxy * expe;
		currentSum = currentSum + mult;
	}

	// fxy = complex<float>(im[index(1, 0, 278)], 0);
	// currentSum = complex<float>(im[1], fxy.real());
	return currentSum;
}

__global__ void sumImageLine(const float* image, const int *w, const int *h, const int *u, const int *v, float *sum)
{
	int x = threadIdx.x;
	complex<float> currentSum = sumCol(image, x, *w, *h, *u, *v);
	
	sum[2 * x] = currentSum.real();
	sum[2 * x + 1] = currentSum.imag();	
}

void host_sumImageLine(const float* image, const int host_w, const int* dev_w, const int* dev_h, const int* dev_u, const int* dev_v, float* dev_sum)
{
	sumImageLine<<<1, host_w>>> (image, dev_w, dev_h, dev_u, dev_v, dev_sum);
}