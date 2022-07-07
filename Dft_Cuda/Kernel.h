#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/complex.h>

__global__ void sumImageLine(const float* image, const int* w, const int* h, const int* u, const int* v, float* sum);
void host_sumImageLine(const float* image, const int host_w, const int* w, const int* h, const int* u, const int* v, float* sum);