import math
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair,_triple
import MDCONV_CUDA

class DeformConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, offset, weight, bias=None, stride=1, padding=0, dilation=1,
                groups=1, deformable_groups=1 , in_step=64):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.in_step = in_step
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(0)  # fake tensor
        if not input.is_cuda:
            raise NotImplementedError
        if weight.requires_grad or offset.requires_grad or input.requires_grad:
            ctx.save_for_backward(input, offset, weight, bias)
        output = input.new_empty(DeformConv2dFunction._infer_shape(ctx, input, weight))

        MDCONV_CUDA.deform_conv2d_forward_cuda(
            input, weight, bias, offset, output,
            weight.shape[2],weight.shape[3],
            ctx.stride[0], ctx.stride[1],
            ctx.padding[0], ctx.padding[1],
            ctx.dilation[0],ctx.dilation[1],
            ctx.groups, ctx.deformable_groups,ctx.in_step, ctx.with_bias)
        '''
        int deform_conv2d_forward_cuda(
            at::Tensor input, at::Tensor weight, at::Tensor bias,
            at::Tensor offset, at::Tensor output,
            const int kernel_h, const int kernel_w, const int stride_h, const int stride_w,
            const int pad_h, const int pad_w, const int dilation_h,
            const int dilation_w, const int group, const int deformable_group,
            const int in_step,const bool with_bias);
        '''
        return output

    @staticmethod
    # @once_differentiabl
    def backward(ctx, grad_output):
        grad_output=grad_output.contiguous()
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        MDCONV_CUDA.deform_conv2d_backward_cuda(
            input, weight, bias, offset,
            grad_input, grad_weight,grad_bias,grad_offset, grad_output,
            weight.shape[2], weight.shape[3],
            ctx.stride[0], ctx.stride[1],
            ctx.padding[0], ctx.padding[1],
            ctx.dilation[0], ctx.dilation[1],
            ctx.groups, ctx.deformable_groups,ctx.in_step,ctx.with_bias)

        '''
        int deform_conv2d_backward_cuda(
            at::Tensor input, at::Tensor weight, at::Tensor bias,at::Tensor offset,
            at::Tensor grad_input, at::Tensor grad_weight, at::Tensor grad_bias,
            at::Tensor grad_offset, at::Tensor grad_output,
            const int kernel_h,const int kernel_w,const int stride_h,const int stride_w,
            const int pad_h,const int pad_w,const int dilation_h,const int dilation_w,
            const int group,const int deformable_group, const int in_step,const bool with_bias);

        '''
        if not ctx.with_bias:
            grad_bias = None

        # print(grad_input,grad_offset,grad_weight,grad_bias)

        return grad_input, grad_offset, grad_weight, grad_bias, None, None, None, None,None,None

    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding[0] - (ctx.dilation[0] *(kernel_h - 1) + 1)) // ctx.stride[0] + 1
        width_out = (width + 2 * ctx.padding[1] - (ctx.dilation[1] *(kernel_w - 1) + 1)) // ctx.stride[1] + 1
        return n, channels_out, height_out, width_out

class ModulatedDeformConv2dFunction(Function):
    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias=None, stride=1, padding=0, dilation=1,
                groups=1, deformable_groups=1 , in_step=64):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.in_step = in_step
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(0)  # fake tensor
        if not input.is_cuda:
            raise NotImplementedError
        if weight.requires_grad or mask.requires_grad or offset.requires_grad  or input.requires_grad:
            ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(ModulatedDeformConv2dFunction._infer_shape(ctx, input, weight))

        MDCONV_CUDA.modulated_deform_conv2d_forward_cuda(
            input, weight, bias, offset, mask, output,
            weight.shape[2],weight.shape[3],
            ctx.stride[0], ctx.stride[1],
            ctx.padding[0], ctx.padding[1],
            ctx.dilation[0],ctx.dilation[1],
            ctx.groups, ctx.deformable_groups,ctx.in_step, ctx.with_bias)
        '''
        int modulated_deform_conv2d_forward_cuda(
            at::Tensor input, at::Tensor weight, at::Tensor bias,
            at::Tensor offset, at::Tensor mask, at::Tensor output,
            const int kernel_h, const int kernel_w, const int stride_h, const int stride_w,
            const int pad_h, const int pad_w, const int dilation_h,
            const int dilation_w, const int group, const int deformable_group,
            const int in_step,const bool with_bias);
        '''
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_output=grad_output.contiguous()
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        MDCONV_CUDA.modulated_deform_conv2d_backward_cuda(
            input, weight, bias, offset, mask,
            grad_input, grad_weight,grad_bias,grad_offset, grad_mask, grad_output,
            weight.shape[2], weight.shape[3],
            ctx.stride[0], ctx.stride[1],
            ctx.padding[0], ctx.padding[1],
            ctx.dilation[0], ctx.dilation[1],
            ctx.groups, ctx.deformable_groups,ctx.in_step,ctx.with_bias)

        '''
        int modulated_deform_conv2d_backward_cuda(
            at::Tensor input, at::Tensor weight, at::Tensor bias,at::Tensor offset, at::Tensor mask,
            at::Tensor grad_input, at::Tensor grad_weight, at::Tensor grad_bias,
            at::Tensor grad_offset, at::Tensor grad_mask, at::Tensor grad_output,
            const int kernel_h,const int kernel_w,const int stride_h,const int stride_w,
            const int pad_h,const int pad_w,const int dilation_h,const int dilation_w,
            const int group,const int deformable_group, const int in_step,const bool with_bias);

        '''
        if not ctx.with_bias:
            grad_bias = None

        return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias, None, None, None, None,None,None)

    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding[0] - (ctx.dilation[0] *(kernel_h - 1) + 1)) // ctx.stride[0] + 1
        width_out = (width + 2 * ctx.padding[1] - (ctx.dilation[1] *(kernel_w - 1) + 1)) // ctx.stride[1] + 1
        return n, channels_out, height_out, width_out

class DeformConv3dFunction(Function):
    @staticmethod
    def forward(ctx, input, offset, weight, bias=None, stride=1, padding=0, dilation=1,
                groups=1, deformable_groups=1 , in_step=64):
        ctx.stride = _triple(stride)
        ctx.padding = _triple(padding)
        ctx.dilation = _triple(dilation)
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.in_step = in_step
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(0)  # fake tensor
        if not input.is_cuda:
            raise NotImplementedError
        if weight.requires_grad or offset.requires_grad or input.requires_grad:
            ctx.save_for_backward(input, offset, weight, bias)
        output = input.new_empty(DeformConv3dFunction._infer_shape(ctx, input, weight))

        MDCONV_CUDA.deform_conv3d_forward_cuda(
            input, weight, bias, offset, output,
            weight.shape[2],weight.shape[3],weight.shape[4],
            ctx.stride[0], ctx.stride[1],ctx.stride[2],
            ctx.padding[0], ctx.padding[1],ctx.padding[2],
            ctx.dilation[0],ctx.dilation[1],ctx.dilation[2],
            ctx.groups, ctx.deformable_groups,ctx.in_step, ctx.with_bias)
        '''
        int deform_conv3d_forward_cuda(
                at::Tensor input, at::Tensor weight,at::Tensor bias,
                at::Tensor offset, at::Tensor output,
                const int kernel_h,const int kernel_w,const int kernel_l,
                const int stride_h,const int stride_w,const int stride_l,
                const int pad_h,const int pad_w,const int pad_l,
                const int dilation_h,const int dilation_w,const int dilation_l,
                const int group,const int deformable_group, const int in_step,const bool with_bias);
        '''
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_output=grad_output.contiguous()
        # print(grad_output)
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        MDCONV_CUDA.deform_conv3d_backward_cuda(
            input, weight, bias, offset,
            grad_input, grad_weight,grad_bias,grad_offset, grad_output,
            weight.shape[2], weight.shape[3],weight.shape[4],
            ctx.stride[0], ctx.stride[1], ctx.stride[2],
            ctx.padding[0], ctx.padding[1], ctx.padding[2],
            ctx.dilation[0], ctx.dilation[1],ctx.dilation[2],
            ctx.groups, ctx.deformable_groups,ctx.in_step,ctx.with_bias)

        '''
        int deform_conv3d_backward_cuda(
            at::Tensor input, at::Tensor weight, at::Tensor bias,at::Tensor offset,
            at::Tensor grad_input, at::Tensor grad_weight, at::Tensor grad_bias,
            at::Tensor grad_offset, at::Tensor grad_output,
            const int kernel_h,const int kernel_w,const int kernel_l,
            const int stride_h,const int stride_w,const int stride_l,
            const int pad_h,const int pad_w,const int pad_l,
            const int dilation_h,const int dilation_w,const int dilation_l,
            const int group, int deformable_group,const int in_step,const bool with_bias) ;
        '''
        if not ctx.with_bias:
            grad_bias = None

        return (grad_input, grad_offset, grad_weight, grad_bias, None, None, None, None,None,None)


    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width,length = input.shape[2:5]
        kernel_h, kernel_w ,kernel_l= weight.shape[2:5]
        height_out = (height + 2 * ctx.padding[0] - (ctx.dilation[0] * (kernel_h - 1) + 1)) // ctx.stride[0] + 1
        width_out = (width + 2 * ctx.padding[1] - (ctx.dilation[1] * (kernel_w - 1) + 1)) // ctx.stride[1] + 1
        length_out = (length + 2 * ctx.padding[2] - (ctx.dilation[2] * (kernel_l - 1) + 1)) // ctx.stride[2] + 1
        return n, channels_out, height_out, width_out, length_out

class ModulatedDeformConv3dFunction(Function):
    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias=None, stride=1, padding=0, dilation=1,
                groups=1, deformable_groups=1 , in_step=64):
        ctx.stride = _triple(stride)
        ctx.padding = _triple(padding)
        ctx.dilation = _triple(dilation)
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.in_step = in_step
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(0)  # fake tensor
        if not input.is_cuda:
            raise NotImplementedError
        if weight.requires_grad or mask.requires_grad or offset.requires_grad  or input.requires_grad:
            ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(ModulatedDeformConv3dFunction._infer_shape(ctx, input, weight))

        MDCONV_CUDA.modulated_deform_conv3d_forward_cuda(
            input, weight, bias, offset, mask, output,
            weight.shape[2],weight.shape[3],weight.shape[4],
            ctx.stride[0], ctx.stride[1], ctx.stride[2],
            ctx.padding[0], ctx.padding[1], ctx.padding[2],
            ctx.dilation[0],ctx.dilation[1],ctx.dilation[2],
            ctx.groups, ctx.deformable_groups,ctx.in_step, ctx.with_bias)
        '''
        int modulated_deform_conv3d_forward_cuda(
            at::Tensor input, at::Tensor weight, at::Tensor bias,
            at::Tensor offset, at::Tensor mask, at::Tensor output,
            const int kernel_h, const int kernel_w, const int kernel_l,
            const int stride_h, const int stride_w, const int stride_l,
            const int pad_h, const int pad_w, const int pad_l,
            const int dilation_h,const int dilation_w, const int dilation_l,
            const int group, const int deformable_group,const int in_step,const bool with_bias);
        '''
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        # print(grad_output)
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        MDCONV_CUDA.modulated_deform_conv3d_backward_cuda(
            input, weight, bias, offset, mask,
            grad_input, grad_weight,grad_bias,grad_offset, grad_mask, grad_output,
            weight.shape[2], weight.shape[3],weight.shape[4],
            ctx.stride[0], ctx.stride[1],ctx.stride[2],
            ctx.padding[0], ctx.padding[1],ctx.padding[2],
            ctx.dilation[0], ctx.dilation[1],ctx.dilation[2],
            ctx.groups, ctx.deformable_groups,ctx.in_step,ctx.with_bias)

        '''
        int modulated_deform_conv3d_backward_cuda(
            at::Tensor input, at::Tensor weight, at::Tensor bias,at::Tensor offset,at::Tensor mask,
            at::Tensor grad_input, at::Tensor grad_weight, at::Tensor grad_bias,
            at::Tensor grad_offset, at::Tensor grad_mask, at::Tensor grad_output,
            const int kernel_h,const int kernel_w,const int kernel_l,
            const int stride_h,const int stride_w,const int stride_l,
            const int pad_h,const int pad_w,const int pad_l,
            const int dilation_h,const int dilation_w,const int dilation_l,
            const int group,const int deformable_group,const int in_step,const bool with_bias) ;
        '''
        if not ctx.with_bias:
            grad_bias = None

        return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias, None, None, None, None,None,None)

    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width,length = input.shape[2:5]
        kernel_h, kernel_w ,kernel_l= weight.shape[2:5]
        height_out = (height + 2 * ctx.padding[0] - (ctx.dilation[0] * (kernel_h - 1) + 1)) // ctx.stride[0] + 1
        width_out = (width + 2 * ctx.padding[1] - (ctx.dilation[1] * (kernel_w - 1) + 1)) // ctx.stride[1] + 1
        length_out = (length + 2 * ctx.padding[2] - (ctx.dilation[2] * (kernel_l - 1) + 1)) // ctx.stride[2] + 1
        return n, channels_out, height_out, width_out, length_out

deform_conv2d = DeformConv2dFunction.apply
modulated_deform_conv2d = ModulatedDeformConv2dFunction.apply
deform_conv3d = DeformConv3dFunction.apply
modulated_deform_conv3d = ModulatedDeformConv3dFunction.apply

class DeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, deformable_groups=1, bias=False,in_step=64):
        super(DeformConv2d, self).__init__()
        assert in_channels % groups == 0, \
            'in_channels {} cannot be divisible by groups {}'.format(
                in_channels, groups)
        assert out_channels % groups == 0, \
            'out_channels {} cannot be divisible by groups {}'.format(
                out_channels, groups)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.in_step=in_step

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
        self.with_bias=bias
        if self.with_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias=None

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.with_bias:
            self.bias.data.fill_(0)

    def forward(self, x, offset):
        return deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding, self.dilation,
                           self.groups, self.deformable_groups,self.in_step)

    # def forward(ctx, input, offset, weight, bias=None, stride=1, padding=0, dilation=1,
    #             groups=1, deformable_groups=1 , in_step=64):

class ModulatedDeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, deformable_groups=1, bias=False,in_step=64):
        super(ModulatedDeformConv2d, self).__init__()
        assert in_channels % groups == 0, \
            'in_channels {} cannot be divisible by groups {}'.format(
                in_channels, groups)
        assert out_channels % groups == 0, \
            'out_channels {} cannot be divisible by groups {}'.format(
                out_channels, groups)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.in_step=in_step

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
        self.with_bias=bias
        if self.with_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias=None

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.with_bias:
            self.bias.data.fill_(0)

    def forward(self, x, offset,mask):
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation,
                           self.groups, self.deformable_groups,self.in_step)

    # def forward(ctx, input, offset, mask, weight, bias=None, stride=1, padding=0, dilation=1,
    #             groups=1, deformable_groups=1 , in_step=64):

class DeformConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, deformable_groups=1, bias=False,in_step=64):
        super(DeformConv3d, self).__init__()
        assert in_channels % groups == 0, \
            'in_channels {} cannot be divisible by groups {}'.format(
                in_channels, groups)
        assert out_channels % groups == 0, \
            'out_channels {} cannot be divisible by groups {}'.format(
                out_channels, groups)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.in_step=in_step

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
        self.with_bias=bias
        if self.with_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias=None

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.with_bias:
            self.bias.data.fill_(0)

    def forward(self, x, offset):
        return deform_conv3d(x, offset, self.weight, self.bias, self.stride, self.padding, self.dilation,
                           self.groups, self.deformable_groups,self.in_step)

    # def forward(ctx, input, offset, weight, bias=None, stride=1, padding=0, dilation=1,
    #             groups=1, deformable_groups=1 , in_step=64):

class ModulatedDeformConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, deformable_groups=1, bias=False,in_step=64):
        super(ModulatedDeformConv3d, self).__init__()
        assert in_channels % groups == 0, \
            'in_channels {} cannot be divisible by groups {}'.format(
                in_channels, groups)
        assert out_channels % groups == 0, \
            'out_channels {} cannot be divisible by groups {}'.format(
                out_channels, groups)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.in_step=in_step

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
        self.with_bias=bias
        if self.with_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias=None

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.with_bias:
            self.bias.data.fill_(0)

    def forward(self, x, offset,mask):
        return modulated_deform_conv3d(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation,
                           self.groups, self.deformable_groups,self.in_step)

    # def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
    #              groups=1, deformable_groups=1, bias=False, in_step=64):

# class DeformConv2d_(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
#                  groups=1, deformable_groups=1, bias=False,in_step=64):
#         super(DeformConv2d_, self).__init__()
#         assert in_channels % groups == 0, \
#             'in_channels {} cannot be divisible by groups {}'.format(
#                 in_channels, groups)
#         assert out_channels % groups == 0, \
#             'out_channels {} cannot be divisible by groups {}'.format(
#                 out_channels, groups)
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = _pair(kernel_size)
#         self.stride = _pair(stride)
#         self.padding = _pair(padding)
#         self.dilation = _pair(dilation)
#         self.groups = groups
#         self.deformable_groups = deformable_groups
#         self.in_step=in_step
#
#         self.weight = nn.Parameter(
#             torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
#         self.with_bias=bias
#         if self.with_bias:
#             self.bias = nn.Parameter(torch.Tensor(out_channels))
#         else:
#             self.bias=None
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         n = self.in_channels
#         for k in self.kernel_size:
#             n *= k
#         stdv = 1. / math.sqrt(n)
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.with_bias:
#             self.bias.data.fill_(0)
#
#     def forward(self, x, offset):
#         return deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding, self.dilation,
#                            self.groups, self.deformable_group,self.in_step)
#
#     # def forward(ctx, input, offset, weight, bias=None, stride=1, padding=0, dilation=1,
#     #             groups=1, deformable_groups=1 , in_step=64):
#
# class ModulatedDeformConv2d_(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
#                  groups=1, deformable_groups=1, bias=False,in_step=64):
#         super(ModulatedDeformConv2d_, self).__init__()
#         assert in_channels % groups == 0, \
#             'in_channels {} cannot be divisible by groups {}'.format(
#                 in_channels, groups)
#         assert out_channels % groups == 0, \
#             'out_channels {} cannot be divisible by groups {}'.format(
#                 out_channels, groups)
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = _pair(kernel_size)
#         self.stride = _pair(stride)
#         self.padding = _pair(padding)
#         self.dilation = _pair(dilation)
#         self.groups = groups
#         self.deformable_groups = deformable_groups
#         self.in_step=in_step
#
#         self.weight = nn.Parameter(
#             torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
#         self.with_bias=bias
#         if self.with_bias:
#             self.bias = nn.Parameter(torch.Tensor(out_channels))
#         else:
#             self.bias=None
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         n = self.in_channels
#         for k in self.kernel_size:
#             n *= k
#         stdv = 1. / math.sqrt(n)
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.with_bias:
#             self.bias.data.fill_(0)
#
#     def forward(self, x, offset,mask):
#         return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation,
#                            self.groups, self.deformable_group,self.in_step)
#
#     # def forward(ctx, input, offset, mask, weight, bias=None, stride=1, padding=0, dilation=1,
#     #             groups=1, deformable_groups=1 , in_step=64):
#
# class DeformConv3d_(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
#                  groups=1, deformable_groups=1, bias=False,in_step=64):
#         super(DeformConv3d_, self).__init__()
#         assert in_channels % groups == 0, \
#             'in_channels {} cannot be divisible by groups {}'.format(
#                 in_channels, groups)
#         assert out_channels % groups == 0, \
#             'out_channels {} cannot be divisible by groups {}'.format(
#                 out_channels, groups)
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = _triple(kernel_size)
#         self.stride = _triple(stride)
#         self.padding = _triple(padding)
#         self.dilation = _triple(dilation)
#         self.groups = groups
#         self.deformable_groups = deformable_groups
#         self.in_step=in_step
#
#         self.weight = nn.Parameter(
#             torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
#         self.with_bias=bias
#         if self.with_bias:
#             self.bias = nn.Parameter(torch.Tensor(out_channels))
#         else:
#             self.bias=None
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         n = self.in_channels
#         for k in self.kernel_size:
#             n *= k
#         stdv = 1. / math.sqrt(n)
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.with_bias:
#             self.bias.data.fill_(0)
#
#     def forward(self, x, offset):
#         return deform_conv3d(x, offset, self.weight, self.bias, self.stride, self.padding, self.dilation,
#                            self.groups, self.deformable_group,self.in_step)
#
#     # def forward(ctx, input, offset, weight, bias=None, stride=1, padding=0, dilation=1,
#     #             groups=1, deformable_groups=1 , in_step=64):
#
# class ModulatedDeformConv3d_(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
#                  groups=1, deformable_groups=1, bias=False,in_step=64):
#         super(ModulatedDeformConv3d_, self).__init__()
#         assert in_channels % groups == 0, \
#             'in_channels {} cannot be divisible by groups {}'.format(
#                 in_channels, groups)
#         assert out_channels % groups == 0, \
#             'out_channels {} cannot be divisible by groups {}'.format(
#                 out_channels, groups)
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = _pair(kernel_size)
#         self.stride = _pair(stride)
#         self.padding = _pair(padding)
#         self.dilation = _pair(dilation)
#         self.groups = groups
#         self.deformable_groups = deformable_groups
#         self.in_step=in_step
#
#         self.weight = nn.Parameter(
#             torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
#         self.with_bias=bias
#         if self.with_bias:
#             self.bias = nn.Parameter(torch.Tensor(out_channels))
#         else:
#             self.bias=None
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         n = self.in_channels
#         for k in self.kernel_size:
#             n *= k
#         stdv = 1. / math.sqrt(n)
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.with_bias:
#             self.bias.data.fill_(0)
#
#     def forward(self, x, offset,mask):
#         return modulated_deform_conv3d(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation,
#                            self.groups, self.deformable_group,self.in_step)
#
#     # def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
#     #              groups=1, deformable_groups=1, bias=False, in_step=64):

class DeformConv2dPack(DeformConv2d):
    def __init__(self, *args, **kwargs):
        super(DeformConv2dPack, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding),
            bias=True)
        self.init_offset()

    def init_offset(self):

        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.conv_offset.weight.data.uniform_(-stdv, stdv)
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        offset = self.conv_offset(x)
        return deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding, self.dilation,
                             self.groups, self.deformable_groups, self.in_step)

class ModulatedDeformConv2dPack(ModulatedDeformConv2d):
    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConv2dPack, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding),
            bias=True)
        self.conv_mask = nn.Conv2d(
            self.in_channels,
            self.deformable_groups  * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding),
            bias=True)
        self.init_offset_mask()

    def init_offset_mask(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.conv_offset.weight.data.uniform_(-stdv, stdv)
        self.conv_offset.bias.data.zero_()
        self.conv_mask.weight.data.uniform_(-stdv, stdv)
        self.conv_mask.bias.data.zero_()

    def forward(self, x):
        offset = self.conv_offset(x)
        mask = self.conv_mask(x)
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation,
                             self.groups, self.deformable_groups, self.in_step)

class DeformConv3dPack(DeformConv3d):
    def __init__(self, *args, **kwargs):
        super(DeformConv3dPack, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Conv3d(
            self.in_channels,
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2],
            kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True)
        self.init_offset()

    def init_offset(self):

        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.conv_offset.weight.data.uniform_(-stdv, stdv)
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        offset = self.conv_offset(x)
        return deform_conv3d(x, offset, self.weight, self.bias, self.stride, self.padding, self.dilation,
                             self.groups, self.deformable_groups, self.in_step)

class ModulatedDeformConv3dPack(ModulatedDeformConv3d):
    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConv3dPack, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Conv3d(
            self.in_channels,
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2],
            kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True)
        self.conv_mask = nn.Conv3d(
            self.in_channels,
            self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2],
            kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True)
        self.init_offset_mask()

    def init_offset_mask(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.conv_offset.weight.data.uniform_(-stdv, stdv)
        self.conv_offset.bias.data.zero_()
        self.conv_mask.weight.data.uniform_(-stdv, stdv)
        self.conv_mask.bias.data.zero_()

    def forward(self, x):
        offset = self.conv_offset(x)
        mask = self.conv_mask(x)
        return modulated_deform_conv3d(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation,
                             self.groups, self.deformable_groups, self.in_step)

