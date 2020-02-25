# modulated-deform-conv
该项目是一个 Pytorch C++ and CUDA Extension,采用C++和Cuda实现了deformable-conv2d,modulated-deformable-conv2d,deformable-conv3d,modulated-deformable-conv3d的forward function和backward function,并在Python中对其进行了包装。
<br />This Project is a Pytorch C++ and CUDA Extension, which implements  the forward function and backward function for deformable-conv2d, modulated-deformable-conv2d, deformable-conv3d, modulated-deformable-conv3d, then encapsulates C++ and CUDA  code into Python Package.

### 安装 Install
* run `pip install modulated-deform-conv`
* or `git clone https://github.com/CHONSPQX/modulated-deform-conv.git`,then `cd modulated-deform-conv` and run `python setup.py install`


#### 要求 Requires
* Python 3
* Pytorch>=1.3
* Linux, gcc版本>=4.9(For Linux, gcc version>=4.9)
* Windows,CUDA版本需要VS版本兼容(For Windows, CUDA version must be compatiable with Visual Studio version)

由于资源有限，目前测试过的环境有(Because of limited resources, only the following environment are tested)
- Ubuntu18.04 , gcc 7.4 , CUDA 10.2 ,Python3.7.4, Pytorch 1.3.1
- Ubuntu18.04 , gcc 7.4 , CUDA 10.2 ,Python3.7.4, Pytorch 1.4.0
- Windows10 , Visual Studio 2017 , CUDA 10.1 ,Python3.7.6, Pytorch 1.4.0

#### 速度优化  Speed Optimization
* `pip download modulated-deform-conv`
解压得到的压缩文件，进入`modulated-deform-conv`，打开`src/config.h`，用户可根据自身显卡情况，设置以下两个变量，获得更快运行速度，然后运行 `python setup.py install`
<br>Unzip the downloaded compressed file, `cd modulated-deform-conv`, then open `src/config.h`,users are recommended to set the following `VARIABLES` to optimize run speed according to their NVIDIA GPU condition, then run `python setup.py install`
	* `const int CUDA_NUM_THREADS`
	* `const int MAX_GRID_NUM`

* 运行时可以通过传递`in_step`参数来优化速度，该变量控制每次并行处理的batch 大小。
<br> Or users can set different `in_step`  value in run time, which controls the batch size of each parallel processing .

### 使用 Use
直接使用C++函数，请`import MDCONV_CUDA`
使用封装后的python类，请`import modulated_deform_conv`
Using C++ functions directly, please  `import MDCONV_CUDA`
Using the packaged function by Python, please `import modulated_deform_conv`

# 文档 Documents
## 1.C++ and CUDA Code
* 文件 Files

|Filename                      |Content                    |
|:--------------------------:| :-----------------------: |
|`config.h`      | **macro&gloabl variables&inline functions**       |
|`deformable_conv.cu`| **MDCONV_CUDA.deform_conv2d_forward_cuda MDCONV_CUDA.deform_conv2d_backward_cuda**  |
|`mdeformable_conv.cu`| **MDCONV_CUDA.modulated_deform_conv2d_forward_cuda MDCONV_CUDA.modulated_deform_conv2d_backward_cuda**  |
|`deformable_conv3d.cu`| **MDCONV_CUDA.deform_conv3d_forward_cuda  MDCONV_CUDA.deform_conv3d_backward_cuda**  |
|`mdeformable_conv3d.cu`| **MDCONV_CUDA.modulated_deform_conv3d_forward_cuda MDCONV_CUDA.modulated_deform_conv2d_backward_cuda**  |
|`utils.cu`| **some code for display debug outputs**  |
|`warp.cpp`| **glue code between C++ and Python**  |

* 变量 Variables

|Variable Name       |       Type              | Introduction |
| :--------------------:  | :-------------------: |:----------------:|
|`kernel_h`| `const int`|first dimension size of the convolution kernel|
|`kernel_w`| `const int`|second dimension size of the convolution kernel|
|`kernel_l`| `const int`|third dimension size of the convolution kernel|
|`stride_h`| `const int`|stride for first dimension|
|`stride_w`| `const int`|stride for second dimension|
|`stride_l`| `const int`|stride for third dimension|
|`pad_h`| `const int`|zero padding for first dimension|
|`pad_w`| `const int`|zero padding for second dimension|
|`pad_l`| `const int`|zero padding for third dimension|
|`dilation_h`| `const int`|dilation rate for first dimension|
|`dilation_w`| `const int`|dilation rate for second dimension|
|`dilation_l`| `const int`|dilation rate for third dimension|
|`group`| `const int`|group of convolution |
|`deformable_group`| `const int`|group of offset and mask |
|`in_step`| `const int`|batch size of each parallel processing|
|`with_bias`| `const bool`|if have bias|
|`input`| `at::Tensor` |`B,I,H,W[,L]`,`I` must be divisible by`group` and ` deformable_group`|
|`grad_input`| `at::Tensor` |`grad_input` must be size like `input` |
|`weight`| `at::Tensor` |`O,I/group,H,W[,L]`，`O`must be divisible by`group`|
|`grad_weight`| `at::Tensor` |`grad_weight` must be size like `weight`|
|`bias`| `at::Tensor` |`[O]`, if `with_bias=true`, `bias` must be non-null|
|`grad_bias`| `at::Tensor` |`grad_bias` must be size like `bias`|
|`offset`| `at::Tensor` |`B,deformable_group*2*kernel_h*kernel_w,H,W` `B,deformable_group*3*kernel_h*kernel_w*kernel_l,H,W,L`|
|`grad_offset`| `at::Tensor` |`grad_offset` must be size like `offset`|
|`mask`| `at::Tensor` |`B,deformable_group*kernel_h*kernel_w,H,W` `B,deformable_group*kernel_h*kernel_w*kernel_l,H,W,L`|
|`grad_mask`| `at::Tensor` |`grad_mask` must be size like `mask`|
|`output`| `at::Tensor` |`B,O,OH,OW[,OL]`|
|`grad_output`| `at::Tensor` |`grad_output` must be size like `output`|

## 2.Python Code

|Class Name                     |Type                    |
|:--------------------------:| :-----------------------: |
|`class DeformConv2dFunction`      | `torch.autograd.Function`       |
|`class ModulatedDeformConv2dFunction`      | `torch.autograd.Function`      |
|`class DeformConv3dFunction`      | `torch.autograd.Function`       |
|`class ModulatedDeformConv3dFunction`      | `torch.autograd.Function`      |
|`class DeformConv2d`      | `torch.nn.Module`       |
|`class ModulatedDeformConv2d`      | `torch.nn.Module`      |
|`class DeformConv3d`      | `torch.nn.Module`       |
|`class ModulatedDeformConv3d`      | `torch.nn.Module`      |

## Author
**Xin Qiao** `qiaoxin182@gmail.com`
+ [github/chonspqx](https://github.com/chonspqx)

## License
Copyright (c) 2020 Xin Qiao
Released under the MIT license
