#include "config.h"


template <typename scalar_t>
__device__ scalar_t modulated_deform_conv2d_im2col_bilinear(
		const scalar_t *bottom_data, const int data_width,
        const int height, const int width, scalar_t h, scalar_t w)
{
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  scalar_t lh = h - h_low;
  scalar_t lw = w - w_low;
  scalar_t hh = 1 - lh, hw = 1 - lw;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = bottom_data[h_low * data_width + w_low];
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];

  scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename scalar_t>
__global__ void modulated_deform_conv2d_im2col_gpu_kernel(
		const int n, const scalar_t *data_im, const scalar_t *data_offset, const scalar_t *data_mask,
        const int height, const int width, const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        const int channel_per_deformable_group,
        const int batch_size, const int num_channels, const int deformable_group,
        const int height_col, const int width_col,
        scalar_t *data_col)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int b_col = (index / width_col / height_col) % batch_size;
    const int c_im = (index / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;
    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

    scalar_t *data_col_ptr = data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    //const float* data_im_ptr = data_im + ((b_col * num_channels + c_im) * height + h_in) * width + w_in;
    const scalar_t *data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width;
    const scalar_t *data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
    const scalar_t *data_mask_ptr = data_mask + (b_col * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i)
      for (int j = 0; j < kernel_w; ++j)
      {
        const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
        const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
        const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
        const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
        const scalar_t mask = data_mask_ptr[data_mask_hw_ptr];
        scalar_t val = static_cast<scalar_t>(0);
        const scalar_t h_im = h_in + i * dilation_h + offset_h;
        const scalar_t w_im = w_in + j * dilation_w + offset_w;
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width){
          val = modulated_deform_conv2d_im2col_bilinear(data_im_ptr, width, height, width, h_im, w_im);
        }
        *data_col_ptr = val * mask;
        data_col_ptr += batch_size * height_col * width_col;
      }
  }
}

void modulated_deform_conv2d_im2col_cuda(
    at::Tensor data_im, at::Tensor data_offset, at::Tensor data_mask,
    const int batch_size, const int channels, const int height_im, const int width_im,
    const int height_col, const int width_col, const int kernel_h, const int kenerl_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int deformable_group, at::Tensor data_col)
{
  // num_axes should be smaller than block size
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * batch_size * height_col * width_col;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_im.scalar_type(), "modulated_deform_conv2d_im2col_gpu_kernel", ([&] {
        const scalar_t *data_im_ = data_im.data<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data<scalar_t>();
        const scalar_t *data_mask_ = data_mask.data<scalar_t>();
        scalar_t *data_col_ = data_col.data<scalar_t>();

        modulated_deform_conv2d_im2col_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, data_im_, data_offset_, data_mask_, height_im, width_im, kernel_h, kenerl_w,
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, channel_per_deformable_group,
            batch_size, channels, deformable_group, height_col, width_col, data_col_);
      }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in modulated_deform_conv2d_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}

int modulated_deform_conv2d_forward_cuda(
    at::Tensor input, at::Tensor weight, at::Tensor bias,
    at::Tensor offset, at::Tensor mask, at::Tensor output,
    const int kernel_h,const int kernel_w,const const int stride_h, const int stride_w,
    const int pad_h, const int pad_w, const int dilation_h,const int dilation_w,
    const int group, const int deformable_group,const int in_step,
    const bool with_bias) {
  AT_CHECK(input.is_contiguous(), "input tensor has to be contiguous");
  AT_CHECK(weight.is_contiguous(), "weight tensor has to be contiguous");
  AT_CHECK(bias.is_contiguous(), "bias tensor has to be contiguous");
  AT_CHECK(offset.is_contiguous(), "offset tensor has to be contiguous");
  AT_CHECK(mask.is_contiguous(), "mask tensor has to be contiguous");
  AT_CHECK(output.is_contiguous(), "output tensor has to be contiguous");

  const int batch = input.size(0);
  const int channels = input.size(1);
  const int height = input.size(2);
  const int width = input.size(3);

  const int channels_out = weight.size(0);
  const int channels_kernel = weight.size(1);
  const int kernel_h_ = weight.size(2);
  const int kernel_w_ = weight.size(3);

  if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
    AT_ERROR("Input shape and kernel shape wont match: (%d x %d vs %d x %d).",
             kernel_h_, kernel_w, kernel_h_, kernel_w_);
  if (channels != channels_kernel * group)
    AT_ERROR("Input shape and kernel channels wont match: (%d vs %d).",
             channels, channels_kernel * group);

  const int height_out =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_out =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  // resize output
  const int step=GET_STEP(batch,in_step);
  output = output.view({batch/step, step, channels_out, height_out, width_out});
  output.zero_();
  // resize temporary columns
  at::Tensor columns =at::zeros({channels * kernel_h * kernel_w,
	  	  	  step * height_out * width_out},input.options());
  input=input.view({batch/step,step,channels,height,width});
  offset=offset.view({batch/step,step,deformable_group * 2 *kernel_h*kernel_w,height_out,width_out});
  mask=mask.view({batch/step,step,deformable_group*kernel_h*kernel_w,height_out,width_out});
#ifdef DEBUG
  print_tensor_size("modulated_deform_conv2d_forward_cuda---output size",output);
  print_tensor_size("modulated_deform_conv2d_forward_cuda---input size",input);
  print_tensor_size("modulated_deform_conv2d_forward_cuda---offset size",offset);
  print_tensor_size("modulated_deform_conv2d_forward_cuda---mask size",mask);
#endif
  // divide into group
  output = output.view({output.size(0), group, output.size(1) / group,
                        output.size(2), output.size(3),output.size(4)});
  weight = weight.view({group, weight.size(0) / group, weight.size(1),
                        weight.size(2), weight.size(3)});
#ifdef DEBUG
  print_tensor_size("modulated_deform_conv2d_forward_cuda---weight size",weight);
#endif
  for (int b = 0; b < batch/step; b++) {
    columns.fill_(0);
    modulated_deform_conv2d_im2col_cuda(
        input[b], offset[b], mask[b], step, channels, height, width, height_out,
        width_out, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, deformable_group, columns);
    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
#ifdef DEBUG
  print_tensor_size("modulated_deform_conv2d_forward_cuda---columns size",columns);
#endif
    for (int g = 0; g < group; g++) {
      at::Tensor temp=at::mm(weight[g].flatten(1), columns[g]);
      output[b][g] += temp.view_as(output[b][g]);
    }
    columns = columns.view({columns.size(0) * columns.size(1), columns.size(2)});
  }
  weight = weight.view({weight.size(0) * weight.size(1), weight.size(2),
                        weight.size(3), weight.size(4)});
  output = output.view({output.size(0), output.size(1) * output.size(2),
                        output.size(3), output.size(4),output.size(5)});
  output = output.view({batch / step, channels_out, step, height_out, width_out});
  output.transpose_(1, 2);
  output = output.contiguous().view({batch , channels_out, height_out, width_out});
  if (with_bias) {
    output += bias.view({1, bias.size(0), 1, 1});
  }
  input=input.view({batch,channels,height,width});
  offset=offset.view({batch,deformable_group * 2 *kernel_h*kernel_w,height_out,width_out});
  mask=mask.view({batch,deformable_group*kernel_h*kernel_w,height_out,width_out});
  return 0;
}

template <typename scalar_t>
__global__ void modulated_deform_conv2d_gradient_gpu_kernel(
		const int n,const scalar_t *grad_col, const scalar_t *data_input,
		const scalar_t *data_offset, const scalar_t *data_mask, scalar_t * columns,
        const int channels_input, const int height_input, const int width_input,
        const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        const int channel_per_deformable_group, const int step,
        const int offset_channels,const int deformable_group,
        const int height_col, const int width_col,
        scalar_t * grad_input,scalar_t *grad_offset, scalar_t *grad_mask)
{
// columns = {channels * kernel_h * kernel_w,step*height_out * width_out}
// grad_columns =  {channels * kernel_h * kernel_w,step*height_out * width_out}
// grad_output = {step,channels_out,height_out,width_out}
// input = {step,channels,height,width}
// offset = {step,deformable_group * 2 * kernel_h * kernel_w,height_out,width_out}
// grad_offset = {step,deformable_group * 2 * kernel_h * kernel_w,height_out,width_out});
  CUDA_KERNEL_LOOP(index, n)//channels*kernel_h * kernel_w * step * height_col * width_col;
  {
	int k = (index /step/ height_col / width_col)%(kernel_h*kernel_w);
	int i=k/kernel_w;
	int j=k%kernel_w;

	int bpos=(index%(step*height_col*width_col))/(height_col*width_col);
    int wpos_col = (index % (height_col*width_col)) % width_col;
    int hpos_col = ((index %(height_col*width_col)) / width_col) % height_col;
    int cpos_col = (index / step / width_col / height_col);
    int cpos_in=cpos_col/kernel_h/kernel_w;
    int offset_group_index=cpos_in/(channels_input/deformable_group);
    //printf("index %d bpos %d cpos_col %d hpos_col %d wpos_col %d \n",index,bpos,cpos_col,hpos_col,wpos_col);
    int offset_h_ptr=bpos*(deformable_group * 2 * kernel_h * kernel_w*height_col*width_col)
    				+offset_group_index*channel_per_deformable_group*height_col*width_col
    				+2*k*height_col*width_col+hpos_col*width_col+wpos_col;
    int offset_w_ptr=bpos*(deformable_group * 2 * kernel_h * kernel_w*height_col*width_col)
    				+offset_group_index*channel_per_deformable_group*height_col*width_col
    				+(2*k+1)*height_col*width_col+hpos_col*width_col+wpos_col;
    int mask_hw_ptr=bpos*(deformable_group * kernel_h * kernel_w*height_col*width_col)
    				+offset_group_index*kernel_h*kernel_w*height_col*width_col
    				+k*height_col*width_col+hpos_col*width_col+wpos_col;
    scalar_t offset_h=data_offset[offset_h_ptr];
    scalar_t offset_w=data_offset[offset_w_ptr];

    int hpos_in = hpos_col * stride_h  -pad_h + (i) * dilation_h;
    int wpos_in = wpos_col * stride_w - pad_w + (j) * dilation_w;
    auto real_offset_h=hpos_in+offset_h;
    auto real_offset_w=wpos_in+offset_w;
    int h_low = floor(real_offset_h);
    int w_low = floor(real_offset_w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;
    scalar_t dh = real_offset_h - h_low;
    scalar_t dw = real_offset_w - w_low;
    scalar_t v1 = 0;
    if (h_low  >= 0 && h_low <= height_input -1  && w_low >= 0  && w_low <= width_input - 1 )
      v1 = data_input[bpos*channels_input*height_input*width_input+cpos_in*height_input*width_input+h_low * width_input + w_low];
    scalar_t v2 = 0;
    if (h_low  >= 0 && h_low <= height_input - 1  && w_high >= 0  && w_high <= width_input - 1&&  abs(dw)>EPS )
      v2 = data_input[bpos*channels_input*height_input*width_input+cpos_in*height_input*width_input+h_low * width_input + w_high];
    scalar_t v3 = 0;
    if (h_high >= 0 && h_high <= height_input - 1 && w_low  >= 0  && w_low <= width_input - 1 && abs(dh)>EPS)
      v3 = data_input[bpos*channels_input*height_input*width_input+cpos_in*height_input*width_input+h_high * width_input + w_low];
    scalar_t v4 = 0;
    if (h_high >= 0 && h_high <= height_input - 1 && w_high >= 0  && w_high <= width_input - 1 && abs(dh)>EPS && abs(dw)>EPS )
      v4 = data_input[bpos*channels_input*height_input*width_input+cpos_in*height_input*width_input+h_high * width_input + w_high];

    scalar_t w1 = (1-dh) *(1- dw), w2 =(1- dh) * dw, w3 = dh*(1- dw), w4 = dh * dw;
    scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    scalar_t col=val*data_mask[mask_hw_ptr];//
    auto dval=data_mask[mask_hw_ptr]*grad_col[index];

    if (h_low >= 0  && h_low<= height_input - 1   && w_low >= 0    && w_low<= width_input - 1)
    	atomicAdd(grad_input + bpos*channels_input*height_input*width_input
    			+ cpos_in*height_input*width_input+h_low * width_input + w_low, w1*dval);
    if (h_low >= 0  && h_low<= height_input - 1   && w_high>=0     && w_high <= width_input - 1)
    	atomicAdd(grad_input + bpos*channels_input*height_input*width_input
    			+ cpos_in*height_input*width_input+h_low * width_input + w_high, w2*dval);
    if (h_high >= 0 && h_high <= height_input - 1 && w_low >= 0    && w_low<= width_input - 1)
    	atomicAdd(grad_input + bpos*channels_input*height_input*width_input
    			+ cpos_in*height_input*width_input+h_high * width_input + w_low, w3*dval);
    if  (h_high >= 0 && h_high <= height_input - 1 && w_high>=0     && w_high <= width_input - 1)
    	atomicAdd(grad_input + bpos*channels_input*height_input*width_input
    			+ cpos_in*height_input*width_input+h_high * width_input + w_high,w4*dval);
    //grad_offset[offset_h_ptr]+=(-1*(1-dw)*v1-1*dw*v2+(1-dw)*v3+dw*v4)*dval;
    atomicAdd(grad_offset + offset_h_ptr,(-1*(1-dw)*v1-1*dw*v2+(1-dw)*v3+dw*v4)*dval);
    //grad_offset[offset_w_ptr]+=(-1*(1-dh)*v1+(1-dh)*v2-1*dh*v3+dh*v4)*dval;
    atomicAdd(grad_offset + offset_w_ptr,(-1*(1-dh)*v1+(1-dh)*v2-1*dh*v3+dh*v4)*dval);
    atomicAdd(grad_mask + mask_hw_ptr,val*grad_col[index]);
    columns[index]=col;
  }
}

// gradient offset mask input
void modulated_deform_conv2d_gradient_cuda(
    at::Tensor grad_col,at::Tensor data_input,
    at::Tensor data_offset, at::Tensor data_mask, at::Tensor columns,
    const int channels, const int height_input, const int width_input,
    const int height_col, const int width_col, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int step, const int deformable_group,
    at::Tensor grad_input, at::Tensor grad_offset, at::Tensor grad_mask)
{
  const int num_kernels =channels*height_col * width_col * kernel_h * kernel_w * step;
  const int channel_per_deformable_group =2 * kernel_h * kernel_w ;
#ifdef DEBUG
  print_tensor_size("modulated_deform_conv2d_gradient_cuda---grad_col size",grad_col,2);
  print_tensor_size("modulated_deform_conv2d_gradient_cuda---data_input size",data_input,2);
  print_tensor_size("modulated_deform_conv2d_gradient_cuda---data_offset size",data_offset,2);
  print_tensor_size("modulated_deform_conv2d_gradient_cuda---data_mask size",data_mask,2);
  print_tensor_size("modulated_deform_conv2d_gradient_cuda---columns size",columns,2);
  print_tensor_size("modulated_deform_conv2d_gradient_cuda---grad_input size",grad_input,2);
  print_tensor_size("modulated_deform_conv2d_gradient_cuda---grad_offset size",grad_offset,2);
  print_tensor_size("modulated_deform_conv2d_gradient_cuda---grad_mask size",grad_mask,2);
#endif

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_col.scalar_type(), "modulated_deform_conv2d_gradient_gpu_kernel", ([&] {
        const scalar_t *grad_col_ = grad_col.data<scalar_t>();
        const scalar_t *data_input_ = data_input.data<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data<scalar_t>();
        const scalar_t *data_mask_ = data_mask.data<scalar_t>();
        scalar_t *columns_ = columns.data<scalar_t>();
        scalar_t *grad_input_ = grad_input.data<scalar_t>();
        scalar_t *grad_offset_ = grad_offset.data<scalar_t>();
        scalar_t *grad_mask_ = grad_mask.data<scalar_t>();
        modulated_deform_conv2d_gradient_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, grad_col_, data_input_, data_offset_, data_mask_,columns_,
            channels, height_input, width_input,
            kernel_h, kernel_w, pad_h, pad_w,
            stride_h, stride_w, dilation_h, dilation_w,
            channel_per_deformable_group,step,
            channel_per_deformable_group * deformable_group,
            deformable_group, height_col, width_col,
            grad_input_,grad_offset_, grad_mask_);
      }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in modulated_deform_conv2d_gradient_cuda: %s\n", cudaGetErrorString(err));
  }
}

int modulated_deform_conv2d_backward_cuda(
    at::Tensor input, at::Tensor weight, at::Tensor bias,at::Tensor offset, at::Tensor mask,
    at::Tensor grad_input, at::Tensor grad_weight, at::Tensor grad_bias,
    at::Tensor grad_offset, at::Tensor grad_mask, at::Tensor grad_output,
    const int kernel_h,const int kernel_w,const int stride_h,const int stride_w,
    const int pad_h,const int pad_w,const int dilation_h,const int dilation_w,
    const int group,const int deformable_group, const int in_step,const bool with_bias) {
  AT_CHECK(input.is_contiguous(), "input tensor has to be contiguous");
  AT_CHECK(weight.is_contiguous(), "weight tensor has to be contiguous");
  AT_CHECK(bias.is_contiguous(), "bias tensor has to be contiguous");
  AT_CHECK(offset.is_contiguous(), "offset tensor has to be contiguous");
  AT_CHECK(mask.is_contiguous(), "mask tensor has to be contiguous");
  AT_CHECK(grad_input.is_contiguous(), "grad_input tensor has to be contiguous");
  AT_CHECK(grad_weight.is_contiguous(), "grad_weight tensor has to be contiguous");
  AT_CHECK(grad_bias.is_contiguous(), "grad_bias tensor has to be contiguous");
  AT_CHECK(grad_offset.is_contiguous(), "grad_offset tensor has to be contiguous");
  AT_CHECK(grad_mask.is_contiguous(), "grad_mask tensor has to be contiguous");
  AT_CHECK(grad_output.is_contiguous(), "grad_output tensor has to be contiguous");

  const int batch = input.size(0);
  const int channels = input.size(1);
  const int height = input.size(2);
  const int width = input.size(3);
  const int channels_out=weight.size(0);
  const int channels_kernel = weight.size(1);
  const int kernel_h_ = weight.size(2);
  const int kernel_w_ = weight.size(3);

  if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
    AT_ERROR("Input shape and kernel shape wont match: (%d x %d vs %d x %d).",
             kernel_h_, kernel_w, kernel_h_, kernel_w_);
  if (channels != channels_kernel * group)
    AT_ERROR("Input shape and kernel channels wont match: (%d vs %d).",
             channels, channels_kernel * group);

  const int height_out =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_out =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int step=GET_STEP(batch,in_step);

  at::Tensor ones = at::ones({step,height_out, width_out}, input.options());
  at::Tensor columns = at::zeros({channels * kernel_h * kernel_w, step*height_out * width_out},input.options());
  at::Tensor grad_columns=at::zeros({channels * kernel_h * kernel_w, step*height_out * width_out},input.options());

  grad_output=grad_output.view({batch/step,step,channels_out,height_out,width_out});
  grad_output.transpose_(1, 2);
  grad_output =grad_output.view({grad_output.size(0), group, grad_output.size(1) / group,
                        grad_output.size(2), grad_output.size(3),grad_output.size(4)});
  input=input.view({batch/step,step,channels,height,width});
  grad_input = grad_input.view({batch/step,step, channels, height, width});
  offset=offset.view({batch/step,step,deformable_group * 2 * kernel_h * kernel_w,height_out,width_out});
  grad_offset=grad_offset.view({batch/step,step,deformable_group * 2 * kernel_h * kernel_w,height_out,width_out});
  mask=mask.view({batch/step,step,deformable_group * kernel_h * kernel_w,height_out,width_out});
  grad_mask=grad_mask.view({batch/step,step,deformable_group * kernel_h * kernel_w,height_out,width_out});
  for (int b = 0; b < batch/step; b++) {
    // divide int group
	grad_columns = grad_columns.view({group, grad_columns.size(0) / group, grad_columns.size(1)});
    weight = weight.view({group, weight.size(0) / group, weight.size(1),weight.size(2), weight.size(3)});
#ifdef DEBUG
  print_tensor_size("modulated_deform_conv2d_backward_cuda_v2---grad_columns size",grad_columns);
  print_tensor_size("modulated_deform_conv2d_backward_cuda_v2---weight size",weight);
#endif
    for (int g = 0; g < group; g++) {
    	grad_columns[g].addmm_(weight[g].flatten(1).transpose(0, 1),grad_output[b][g].flatten(1), 0.0f, 1.0f);
    }
    grad_columns = grad_columns.view({grad_columns.size(0) * grad_columns.size(1), grad_columns.size(2)});
    weight = weight.view({weight.size(0) * weight.size(1), weight.size(2),
                          weight.size(3), weight.size(4)});
    columns.fill_(0);
    modulated_deform_conv2d_gradient_cuda(
    		grad_columns,input[b],offset[b],mask[b],columns,
    		channels,height,width,height_out,width_out,kernel_h,kernel_w,
    		pad_h,pad_w,stride_h,stride_w,dilation_h, dilation_w,
    		step,deformable_group,
    		grad_input[b],grad_offset[b],grad_mask[b]);
    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
   // std::cout<<"columns \n"<<columns<<std::endl;
    grad_weight = grad_weight.view({group, grad_weight.size(0) / group,
                                    grad_weight.size(1), grad_weight.size(2),
                                    grad_weight.size(3)});
    if (with_bias)
      grad_bias = grad_bias.view({group, grad_bias.size(0) / group});
    for (int g = 0; g < group; g++) {
      grad_weight[g] =	grad_weight[g].flatten(1)
          	  	  	  	.addmm_(grad_output[b][g].flatten(1), columns[g].transpose(0, 1),1.0f,1.0f)
          	  	  	  	.view_as(grad_weight[g]);
      if (with_bias) {
      	at::Tensor temp=grad_bias[g].view({-1, 1});
      	temp.addmm_(grad_output[b][g].flatten(1), ones.view({-1, 1}),1.0f,1.0f);
        grad_bias[g] =temp.view(-1);
      }
    }
    columns = columns.view({columns.size(0) * columns.size(1), columns.size(2)});
    grad_weight = grad_weight.view({grad_weight.size(0) * grad_weight.size(1),
                                    grad_weight.size(2), grad_weight.size(3),
                                    grad_weight.size(4)});
    if (with_bias)
      grad_bias = grad_bias.view({grad_bias.size(0) * grad_bias.size(1)});
  }
  // batch/step group channels/group step h w
  grad_output = grad_output.view({grad_output.size(0) ,grad_output.size(1)*grad_output.size(2),
	  	  	  	  	  	  	  	  grad_output.size(3),grad_output.size(4),grad_output.size(5)});
  //grad_output=grad_output.view({batch/step,channels_kernel,step,height_out,width_out});
  grad_output.transpose_(1, 2);
  grad_output =grad_output.view({batch,channels_out,height_out,width_out});
  input=input.view({batch,channels,height,width});
  grad_input = grad_input.view({batch, channels, height, width});
  offset=offset.view({batch,deformable_group * 2 * kernel_h * kernel_w,height_out,width_out});
  grad_offset=grad_offset.view({batch,deformable_group * 2 * kernel_h * kernel_w,height_out,width_out});
  mask=mask.view({batch,deformable_group * kernel_h * kernel_w,height_out,width_out});
  grad_mask=grad_mask.view({batch,deformable_group * kernel_h * kernel_w,height_out,width_out});
  return 0;
}




