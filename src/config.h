/*
 * config.h
 *
 *  Created on: 2020年2月7日
 *      Author: chonsp
 */
#ifndef CONFIG_H_
#define CONFIG_H_
#include<iostream>
#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

using namespace at;

#define EPS 1e-8
#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 256;
const int MAX_GRID_NUM = 65535;

inline int GET_BLOCKS(const int N)
{
  return std::min(MAX_GRID_NUM, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
}

inline int GET_STEP(const int batch,const int step)
{
	int max ,min,temp;
	if (batch >step){
		max = batch;
		min = step;
	}
	else{
		max = step;
		min = batch;
	}
	while (max%min != 0){
		temp = max%min;
		max = min;
		min = temp;
	}
	return min;
}

#define INFO_CHECK(n,msg, ...)    \
	if(blockIdx.x * blockDim.x + threadIdx.x==n){ \
		printf(msg,__VA_ARGS__);    \
	}

#define GETTYPE(instr) #instr


template <typename T>
std::string TO_STRING(T msg)
{
	std::string str;
	std::stringstream conv;
	conv<<msg;
	conv>>str;
	return str;
}
void print_tensor_size(const std::string& info,const at::Tensor data,int level=1);

void function_info(const std::string& info,int level=0);

#define CONDITION_PRINT(n, ...)    \
	if(blockIdx.x * blockDim.x + threadIdx.x==n){ \
		__VA_ARGS__;    \
	}

#define CONDITIONED_PRINT(msg, ...)    \
	if(msg){ \
		__VA_ARGS__;    \
	}

int deform_conv2d_forward_cuda(
    at::Tensor input, at::Tensor weight, at::Tensor bias,
    at::Tensor offset, at::Tensor output,
    const int kernel_h, const int kernel_w, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int group, const int deformable_group,
    const int in_step,const bool with_bias);

int deform_conv2d_backward_cuda(
    at::Tensor input, at::Tensor weight, at::Tensor bias,at::Tensor offset,
    at::Tensor grad_input, at::Tensor grad_weight, at::Tensor grad_bias,
    at::Tensor grad_offset, at::Tensor grad_output,
    const int kernel_h,const int kernel_w,const int stride_h,const int stride_w,
    const int pad_h,const int pad_w,const int dilation_h,const int dilation_w,
    const int group,const int deformable_group, const int in_step,const bool with_bias);


int modulated_deform_conv2d_forward_cuda(
    at::Tensor input, at::Tensor weight, at::Tensor bias,
    at::Tensor offset, at::Tensor mask, at::Tensor output,
    const int kernel_h, const int kernel_w, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int group, const int deformable_group,
    const int in_step,const bool with_bias);

int modulated_deform_conv2d_backward_cuda(
    at::Tensor input, at::Tensor weight, at::Tensor bias,at::Tensor offset, at::Tensor mask,
    at::Tensor grad_input, at::Tensor grad_weight, at::Tensor grad_bias,
    at::Tensor grad_offset, at::Tensor grad_mask, at::Tensor grad_output,
    const int kernel_h,const int kernel_w,const int stride_h,const int stride_w,
    const int pad_h,const int pad_w,const int dilation_h,const int dilation_w,
    const int group,const int deformable_group, const int in_step,const bool with_bias);

int deform_conv3d_forward_cuda(
		at::Tensor input, at::Tensor weight,at::Tensor bias,
        at::Tensor offset, at::Tensor output,
        const int kernel_h,const int kernel_w,const int kernel_l,
        const int stride_h,const int stride_w,const int stride_l,
        const int pad_h,const int pad_w,const int pad_l,
        const int dilation_h,const int dilation_w,const int dilation_l,
        const int group,const int deformable_group, const int in_step,const bool with_bias);

int deform_conv3d_backward_cuda(
    at::Tensor input, at::Tensor weight, at::Tensor bias,at::Tensor offset,
    at::Tensor grad_input, at::Tensor grad_weight, at::Tensor grad_bias,
    at::Tensor grad_offset, at::Tensor grad_output,
    const int kernel_h,const int kernel_w,const int kernel_l,
    const int stride_h,const int stride_w,const int stride_l,
    const int pad_h,const int pad_w,const int pad_l,
    const int dilation_h,const int dilation_w,const int dilation_l,
    const int group, int deformable_group,const int in_step,const bool with_bias) ;

int modulated_deform_conv3d_forward_cuda(
    at::Tensor input, at::Tensor weight, at::Tensor bias,
    at::Tensor offset, at::Tensor mask, at::Tensor output,
    const int kernel_h, const int kernel_w, const int kernel_l,
    const int stride_h, const int stride_w, const int stride_l,
    const int pad_h, const int pad_w, const int pad_l,
    const int dilation_h,const int dilation_w, const int dilation_l,
    const int group, const int deformable_group,const int in_step,const bool with_bias);

int modulated_deform_conv3d_backward_cuda(
    at::Tensor input, at::Tensor weight, at::Tensor bias,at::Tensor offset,at::Tensor mask,
    at::Tensor grad_input, at::Tensor grad_weight, at::Tensor grad_bias,
    at::Tensor grad_offset, at::Tensor grad_mask, at::Tensor grad_output,
    const int kernel_h,const int kernel_w,const int kernel_l,
    const int stride_h,const int stride_w,const int stride_l,
    const int pad_h,const int pad_w,const int pad_l,
    const int dilation_h,const int dilation_w,const int dilation_l,
    const int group,const int deformable_group,const int in_step,const bool with_bias) ;


#endif /* CONFIG_H_ */
