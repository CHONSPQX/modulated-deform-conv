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
#include <torch/extension.h>
#include <pybind11/pybind11.h>

using namespace at;
namespace py = pybind11;

#define EPS 1.192092896e-07F
#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)
      
#define TRY_ALLOCATE(left , right)  \
    try  {                     \
        left = right ;              \
    }catch(std::exception e){  \
        std::cout<<#left<<" TRY_ALLOCATE failed"<<std::endl;          \
        throw e;               \
    }                          
    
      
const int CUDA_NUM_THREADS = 256;
const int MAX_GRID_NUM = 65535;

// const int CUDA_NUM_THREADS = 1;
// const int MAX_GRID_NUM = 1;

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

#ifndef AT_CHECK
#define AT_CHECK TORCH_CHECK 
#endif

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

#endif /* CONFIG_H_ */
