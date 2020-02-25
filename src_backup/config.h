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
        at::Tensor input, at::Tensor weight,
        at::Tensor offset, at::Tensor output,
        int kH, int kW, int dH, int dW, int padH, int padW,
        int dilationH, int dilationW, int group,
        int deformable_group, int im2col_step);

int deform_conv2d_backward_input_cuda(
		at::Tensor input, at::Tensor offset,
        at::Tensor gradOutput, at::Tensor gradInput,
        at::Tensor gradOffset, at::Tensor weight,
        int kH, int kW, int dH,int dW, int padH, int padW, int dilationH,int dilationW,
        int group,int deformable_group, int im2col_step) ;

int deform_conv2d_backward_parameters_cuda(
        at::Tensor input, at::Tensor offset, at::Tensor gradOutput,at::Tensor gradWeight,
        int kH, int kW, int dH, int dW,int padH, int padW, int dilationH, int dilationW,
        int group,int deformable_group, int im2col_step);

int deform_conv2d_backward_cuda_v2(
        at::Tensor input, at::Tensor weight, at::Tensor bias,at::Tensor offset,
        at::Tensor grad_input, at::Tensor grad_weight, at::Tensor grad_bias,
        at::Tensor grad_offset, at::Tensor grad_output,
        int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
        int pad_w, int dilation_h, int dilation_w, int group, int deformable_group,
        const bool with_bias,int im2col_step) ;

int deform_conv3d_forward_cuda(
        at::Tensor input, at::Tensor weight,
        at::Tensor offset, at::Tensor output,
        int kW,int kH, int kL,int dW, int dH, int dL,int padW, int padH,int padL,
        int dilationW, int dilationH, int dilationL,int group,
        int im2col_step);

int deform_conv3d_backward_cuda(
        at::Tensor input, at::Tensor weight, at::Tensor bias,at::Tensor offset,
        at::Tensor grad_input, at::Tensor grad_weight, at::Tensor grad_bias,
        at::Tensor grad_offset, at::Tensor grad_output,
        int kernel_h, int kernel_w, int kernel_l, int stride_h, int stride_w, int stride_l,
        int pad_h, int pad_w, int pad_l, int dilation_h, int dilation_w, int dilation_l,
        int group, int deformable_group,const bool with_bias);

int modulated_deform_conv2d_forward_cuda(
        at::Tensor input, at::Tensor weight, at::Tensor bias,
        at::Tensor offset, at::Tensor mask, at::Tensor output,
        int kernel_h, int kernel_w, const int stride_h, const int stride_w,
        const int pad_h, const int pad_w, const int dilation_h,
        const int dilation_w, const int group, const int deformable_group,
        const bool with_bias);

int modulated_deform_conv2d_backward_cuda(
        at::Tensor input, at::Tensor weight, at::Tensor bias,
        at::Tensor offset, at::Tensor mask,
        at::Tensor grad_input, at::Tensor grad_weight, at::Tensor grad_bias,
        at::Tensor grad_offset, at::Tensor grad_mask, at::Tensor grad_output,
        int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
        int pad_w, int dilation_h, int dilation_w, int group, int deformable_group,
        const bool with_bias);

int modulated_deform_conv2d_backward_cuda_v2(
        at::Tensor input, at::Tensor weight, at::Tensor bias,
        at::Tensor offset, at::Tensor mask,
        at::Tensor grad_input, at::Tensor grad_weight, at::Tensor grad_bias,
        at::Tensor grad_offset, at::Tensor grad_mask, at::Tensor grad_output,
        int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
        int pad_w, int dilation_h, int dilation_w, int group, int deformable_group,
        const bool with_bias);

int modulated_deform_conv3d_forward_cuda(
        at::Tensor input, at::Tensor weight, at::Tensor bias,
        at::Tensor offset, at::Tensor mask, at::Tensor output,
        const int kernel_h, const int kernel_w, const int kernel_l,
        const int stride_h, const int stride_w, const int stride_l,
        const int pad_h, const int pad_w, const int pad_l,
        const int dilation_h,const int dilation_w, const int dilation_l,
        const int group, const int deformable_group,const bool with_bias);

int modulated_deform_conv3d_backward_cuda(
        at::Tensor input, at::Tensor weight, at::Tensor bias,at::Tensor offset, at::Tensor mask,
        at::Tensor grad_input, at::Tensor grad_weight, at::Tensor grad_bias,
        at::Tensor grad_offset, at::Tensor grad_mask, at::Tensor grad_output,
        int kernel_h, int kernel_w, int kernel_l, int stride_h, int stride_w, int stride_l,
        int pad_h, int pad_w, int pad_l, int dilation_h, int dilation_w, int dilation_l,
        int group, int deformable_group,const bool with_bias) ;


#endif /* CONFIG_H_ */
