#include <torch/extension.h>

//1
int deform_conv2d_forward_cuda(
        at::Tensor input, at::Tensor weight,
        at::Tensor offset, at::Tensor output,
        int kH, int kW, int dH, int dW, int padH, int padW,
        int dilationH, int dilationW, int group,
        int deformable_group, int im2col_step);
//2
int deform_conv2d_backward_input_cuda(
		at::Tensor input, at::Tensor offset,
        at::Tensor gradOutput, at::Tensor gradInput,
        at::Tensor gradOffset, at::Tensor weight,
        int kH, int kW, int dH,int dW, int padH, int padW, int dilationH,int dilationW,
        int group,int deformable_group, int im2col_step) ;
//3
int deform_conv2d_backward_parameters_cuda(
        at::Tensor input, at::Tensor offset, at::Tensor gradOutput,
        at::Tensor gradWeight,
        int kH, int kW, int dH, int dW,int padH, int padW, int dilationH, int dilationW,
        int group,int deformable_group, int im2col_step);
//4
int deform_conv2d_backward_cuda_v2(
        at::Tensor input, at::Tensor weight, at::Tensor bias,at::Tensor offset,
        at::Tensor grad_input, at::Tensor grad_weight, at::Tensor grad_bias,
        at::Tensor grad_offset, at::Tensor grad_output,
        int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
        int pad_w, int dilation_h, int dilation_w, int group, int deformable_group,
        const bool with_bias,int im2col_step) ;
//5
int deform_conv3d_forward_cuda(
        at::Tensor input, at::Tensor weight,
        at::Tensor offset, at::Tensor output,
        int kW,int kH, int kL,int dW, int dH, int dL,int padW, int padH,int padL,
        int dilationW, int dilationH, int dilationL,int group,
        int im2col_step);
//6
int deform_conv3d_backward_cuda(
        at::Tensor input, at::Tensor weight, at::Tensor bias,at::Tensor offset,
        at::Tensor grad_input, at::Tensor grad_weight, at::Tensor grad_bias,
        at::Tensor grad_offset, at::Tensor grad_output,
        int kernel_h, int kernel_w, int kernel_l, int stride_h, int stride_w, int stride_l,
        int pad_h, int pad_w, int pad_l, int dilation_h, int dilation_w, int dilation_l,
        int group, int deformable_group,const bool with_bias);
//7
int modulated_deform_conv2d_forward_cuda(
        at::Tensor input, at::Tensor weight, at::Tensor bias,
        at::Tensor offset, at::Tensor mask, at::Tensor output,
        int kernel_h, int kernel_w, const int stride_h, const int stride_w,
        const int pad_h, const int pad_w, const int dilation_h,
        const int dilation_w, const int group, const int deformable_group,
        const bool with_bias);
//8
int modulated_deform_conv2d_backward_cuda(
        at::Tensor input, at::Tensor weight, at::Tensor bias,
        at::Tensor offset, at::Tensor mask,
        at::Tensor grad_input, at::Tensor grad_weight, at::Tensor grad_bias,
        at::Tensor grad_offset, at::Tensor grad_mask, at::Tensor grad_output,
        int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
        int pad_w, int dilation_h, int dilation_w, int group, int deformable_group,
        const bool with_bias);
//9
int modulated_deform_conv2d_backward_cuda_v2(
        at::Tensor input, at::Tensor weight, at::Tensor bias,
        at::Tensor offset, at::Tensor mask,
        at::Tensor grad_input, at::Tensor grad_weight, at::Tensor grad_bias,
        at::Tensor grad_offset, at::Tensor grad_mask, at::Tensor grad_output,
        int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
        int pad_w, int dilation_h, int dilation_w, int group, int deformable_group,
        const bool with_bias);
//10
int modulated_deform_conv3d_forward_cuda(
        at::Tensor input, at::Tensor weight, at::Tensor bias,
        at::Tensor offset, at::Tensor mask, at::Tensor output,
        const int kernel_h, const int kernel_w, const int kernel_l,
        const int stride_h, const int stride_w, const int stride_l,
        const int pad_h, const int pad_w, const int pad_l,
        const int dilation_h,const int dilation_w, const int dilation_l,
        const int group, const int deformable_group,const bool with_bias);
//11
int modulated_deform_conv3d_backward_cuda(
        at::Tensor input, at::Tensor weight, at::Tensor bias,at::Tensor offset, at::Tensor mask,
        at::Tensor grad_input, at::Tensor grad_weight, at::Tensor grad_bias,
        at::Tensor grad_offset, at::Tensor grad_mask, at::Tensor grad_output,
        int kernel_h, int kernel_w, int kernel_l, int stride_h, int stride_w, int stride_l,
        int pad_h, int pad_w, int pad_l, int dilation_h, int dilation_w, int dilation_l,
        int group, int deformable_group,const bool with_bias) ;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("deform_conv2d_forward_cuda", &deform_conv2d_forward_cuda,
        "deform_conv2d_forward_cuda");
  m.def("deform_conv2d_backward_input_cuda", &deform_conv2d_backward_input_cuda,
        "deform_conv2d_backward_input_cuda");
  m.def("deform_conv2d_backward_parameters_cuda", &deform_conv2d_backward_parameters_cuda,
        "deform_conv2d_backward_parameters_cuda");
  m.def("deform_conv2d_backward_cuda_v2", &deform_conv2d_backward_cuda_v2,
        "deform_conv2d_backward_cuda_v2");
  m.def("deform_conv3d_forward_cuda", &deform_conv3d_forward_cuda,
        "deform_conv3d_forward_cuda");
  m.def("deform_conv3d_backward_cuda", &deform_conv3d_backward_cuda,
        "deform_conv3d_backward_cuda");

  m.def("modulated_deform_conv2d_forward_cuda", &modulated_deform_conv2d_forward_cuda,
        "modulated_deform_conv2d_forward_cuda");
  m.def("modulated_deform_conv2d_backward_cuda", &modulated_deform_conv2d_backward_cuda,
        "modulated_deform_conv2d_backward_cuda");
  m.def("modulated_deform_conv2d_backward_cuda_v2", &modulated_deform_conv2d_backward_cuda_v2,
        "modulated_deform_conv2d_backward_cuda_v2");
  m.def("modulated_deform_conv3d_forward_cuda", &modulated_deform_conv3d_forward_cuda,
        "modulated_deform_conv3d_forward_cuda");
  m.def("modulated_deform_conv3d_backward_cuda", &modulated_deform_conv3d_backward_cuda,
        "modulated_deform_conv3d_backward_cuda");
}
