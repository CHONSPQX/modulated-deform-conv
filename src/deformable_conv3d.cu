#include "config.h"

template <typename scalar_t>
__device__ scalar_t deform_conv3d_im2col_trilinear(
		const scalar_t *bottom_data, const int data_width,const int data_length,
		const int height, const int width, const int length,scalar_t h, scalar_t w,scalar_t l)
{

  int h_low = floor(h);
  int w_low = floor(w);
  int l_low = floor(l);
  int h_high = h_low + 1;
  int w_high = w_low + 1;
  int l_high = l_low + 1;

  scalar_t lh = h - h_low;//dh
  scalar_t lw = w - w_low;//dw
  scalar_t ll = l - l_low;//dl
  scalar_t hh = 1 - lh, hw = 1 - lw, hl = 1 - ll; //1-dh 1-dw 1-dl

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0 && l_low >= 0)
    v1 = bottom_data[h_low * data_width*data_length + w_low*data_length+ l_low];
  scalar_t v2 = 0;
  if (h_low >= 0 && w_low >=0 && l_high<= length -1)
    v2 = bottom_data[h_low * data_width*data_length + w_low*data_length+ l_high];
  scalar_t v3 = 0;
  if (h_low >= 0 && w_high <= width - 1 && l_low  >= 0)
    v3 = bottom_data[h_low * data_width*data_length + w_high*data_length+ l_low];
  scalar_t v4 = 0;
  if (h_low >= 0 && w_high <= width - 1 && l_high<= length -1)
    v4 = bottom_data[h_low * data_width*data_length + w_high*data_length+ l_high];

  scalar_t v5 = 0;
  if (h_high <= height -1 && w_low >= 0 && l_low >= 0)
    v5 = bottom_data[h_high * data_width*data_length + w_low*data_length+ l_low];
  scalar_t v6 = 0;
  if (h_high <= height -1 && w_low >= 0  && l_high<= length -1)
    v6 = bottom_data[h_high * data_width*data_length + w_low*data_length+ l_high];
  scalar_t v7 = 0;
  if (h_high <= height -1 && w_high <= width - 1 && l_low >= 0)
    v7 = bottom_data[h_high * data_width*data_length + w_high*data_length+ l_low];
  scalar_t v8 = 0;
  if (h_high <= height -1 && w_high <= width - 1 && l_high<= length -1)
    v8 = bottom_data[h_high * data_width*data_length + w_high*data_length+ l_high];

  scalar_t w1 = hh * hw *hl, w2 = hh *hw *ll, w3 = hh * lw*hl, w4 = hh * lw* ll;
  scalar_t w5 = lh * hw *hl, w6 = lh *hw *ll, w7 = lh * lw*hl, w8 = lh * lw* ll;

  scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4+w5 * v5 + w6 * v6 + w7 * v7 + w8 * v8);
  return val;
}

template <typename scalar_t>
__global__ void deform_conv3d_im2col_gpu_kernel(
		const int n, const scalar_t *data_im, const scalar_t *data_offset,
        const int height, const int width, const int length,
        const int kernel_h, const int kernel_w, const int kernel_l,
        const int pad_h, const int pad_w, const int pad_l,
        const int stride_h, const int stride_w, const int stride_l,
        const int dilation_h, const int dilation_w, const int dilation_l,
        const int channel_per_deformable_group,
        const int batch_size, const int num_channels, const int deformable_group,
        const int height_col, const int width_col, const int length_col,
         scalar_t *data_col)
{
	//  input = {step, channels,height, width,length}
	//  offset={step,deformable_group*3* kernel_h * kernel_w * kernel_l,height_out, width_out,length_out}
  CUDA_KERNEL_LOOP(index, n)//num_kernels = channels * batch_size * height_col * width_col *length_col;
  {
//  columns = {channels * kernel_h * kernel_w * kernel_l,step * height_out * width_out * length_out}
	const int l_col=  index % length_col;
	const int w_col = (index / length_col) % width_col;
    const int h_col = (index / length_col / width_col ) % height_col;
    const int b_col = (index / length_col / width_col / height_col) % batch_size;

    const int c_im = (index / length_col / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w * kernel_l;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;
    const int l_in = l_col * stride_l - pad_l;

    scalar_t *data_col_ptr = data_col+ (c_col* batch_size + b_col)* height_col* width_col * length_col
    								 + h_col * width_col * length_col
    								 + w_col * length_col
    								 + l_col; //27 10000
    const scalar_t *data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width * length;
    const scalar_t *data_offset_ptr = data_offset
    		+(b_col * deformable_group + deformable_group_index)
    		* 3 * kernel_h * kernel_w * kernel_l * height_col * width_col * length_col;// 1 18 100 100

    for (int i = 0; i < kernel_h; ++i)
      for (int j = 0; j < kernel_w; ++j)
    	for (int k = 0; k < kernel_l; ++k)
    	{
			int f=i*kernel_w*kernel_l + j*kernel_l+k;
			const int data_offset_h_ptr = (3*f) * height_col * width_col * length_col+ h_col* width_col * length_col+ w_col* length_col + l_col;
			const int data_offset_w_ptr = (3*f+1) * height_col * width_col* length_col + h_col* width_col* length_col + w_col* length_col + l_col;
			const int data_offset_l_ptr = (3*f+2) * height_col * width_col* length_col + h_col* width_col* length_col + w_col* length_col + l_col;
			const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
			const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
			const scalar_t offset_l = data_offset_ptr[data_offset_l_ptr];
			scalar_t val = static_cast<scalar_t>(0);
			const scalar_t h_im = h_in + i * dilation_h + offset_h;
			const scalar_t w_im = w_in + j * dilation_w + offset_w;
			const scalar_t l_im = l_in + k * dilation_l + offset_l;
			if (h_im > -1 && w_im > -1 && l_im > -1 && h_im < height && w_im < width && l_im < length)
			{
			  val = deform_conv3d_im2col_trilinear(data_im_ptr, width, length, height, width, length, h_im, w_im,l_im);
			}
			*data_col_ptr = val;
			data_col_ptr += batch_size * height_col * width_col* length_col;
    	}
  }
}

void deform_conv3d_im2col_cuda(
    at::Tensor data_im, at::Tensor data_offset,
    const int batch_size, const int channels,
    const int height_im, const int width_im, const int length_im,
    const int height_col, const int width_col,const int length_col,
    const int kernel_h,const int kenerl_w,const int kenerl_l,
	const int pad_h, const int pad_w, const int pad_l,
	const int stride_h, const int stride_w, const int stride_l,
	const int dilation_h, const int dilation_w, const int dilation_l,
	const int deformable_group, at::Tensor data_col)
{
	  // num_axes should be smaller than block size
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * batch_size * height_col * width_col *length_col;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_im.scalar_type(), "deform_conv3d_im2col_gpu_kernel", ([&] {
        const scalar_t *data_im_ = data_im.data_ptr<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
        scalar_t *data_col_ = data_col.data_ptr<scalar_t>();

        deform_conv3d_im2col_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, data_im_, data_offset_,
            height_im, width_im, length_im,
            kernel_h, kenerl_w,kenerl_l,
            pad_h, pad_w, pad_l,
            stride_h, stride_w,stride_l,
            dilation_h, dilation_w, dilation_l,
            channel_per_deformable_group, batch_size,
            channels, deformable_group,
            height_col, width_col, length_col,data_col_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in deform_conv3d_im2col: %s\n", cudaGetErrorString(err));
  }
}

int deform_conv3d_forward_cuda(
		at::Tensor input, at::Tensor weight,at::Tensor bias,
        at::Tensor offset, at::Tensor output,
        const int kernel_h,const int kernel_w,const int kernel_l,
        const int stride_h,const int stride_w,const int stride_l,
        const int pad_h,const int pad_w,const int pad_l,
        const int dilation_h,const int dilation_w,const int dilation_l,
        const int group,const int deformable_group, const int in_step,const bool with_bias){
#define DEBUG
  TORCH_CHECK(input.is_contiguous(), "input tensor has to be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight tensor has to be contiguous");
  TORCH_CHECK(bias.is_contiguous(), "bias tensor has to be contiguous");
  TORCH_CHECK(offset.is_contiguous(), "offset tensor has to be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output tensor has to be contiguous");

  const int batch = input.size(0);
  const int channels = input.size(1);
  const int height = input.size(2);
  const int width = input.size(3);
  const int length = input.size(4);

  const int channels_out = weight.size(0);
  const int channels_kernel = weight.size(1);
  const int kernel_h_ = weight.size(2);
  const int kernel_w_ = weight.size(3);
  const int kernel_l_ = weight.size(4);

  if (kernel_h_ != kernel_h || kernel_w_ != kernel_w || kernel_l_ != kernel_l)
    AT_ERROR("Input shape and kernel shape wont match: (%d x %d x %d vs %d x %d x %d).",
             kernel_h_, kernel_w,kernel_l, kernel_h_, kernel_w_,kernel_l_);
  if (channels != channels_kernel * group)
    AT_ERROR("Input shape and kernel channels wont match: (%d vs %d).",
             channels, channels_kernel * group);

  const int height_out =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_out =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int length_out =
      (length + 2 * pad_l - (dilation_l * (kernel_l - 1) + 1)) / stride_l + 1;

  const int step=GET_STEP(batch,in_step);

#ifdef DEBUG
  print_tensor_size("deform_conv3d_forward_cuda---output size",output);
  print_tensor_size("deform_conv3d_forward_cuda---input size",input);
  print_tensor_size("deform_conv3d_forward_cuda---offset size",offset);
#endif

  output = output.view({batch / step, step, channels_out,height_out, width_out,length_out});
  output.zero_();

  at::Tensor columns = at::zeros({channels * kernel_h * kernel_w * kernel_l,
	  step * height_out * width_out * length_out},input.options());

  input = input.view({batch / step, step, channels,height, width,length});
  offset =offset.view({batch / step, step,deformable_group*3* kernel_h * kernel_w * kernel_l,
	  	  	  	  	  height_out, width_out,length_out});
  //divide into group
  output = output.view({output.size(0), group, output.size(1) / group,
                        output.size(2), output.size(3),output.size(4),output.size(5)});
  weight = weight.view({group, weight.size(0) / group, weight.size(1),
	  weight.size(2), weight.size(3),weight.size(4)});

  for (int b = 0; b < batch / step; b++) {
      columns.fill_(0);
	  deform_conv3d_im2col_cuda(
			  input[b], offset[b], step,channels,
			  height,width,length,
			  height_out, width_out, length_out,
			  kernel_h, kernel_w,kernel_l,
			  pad_h, pad_w,pad_l,
			  stride_h, stride_w,stride_l,
			  dilation_h, dilation_w, dilation_l,
			  deformable_group, columns);
	  columns = columns.view({group, columns.size(0) / group, columns.size(1)});
	  for (int g = 0; g < group; g++) {
	      at::Tensor temp=at::mm(weight[g].flatten(1), columns[g]);
	      output[b][g] += temp.view_as(output[b][g]);
	  }
	  columns = columns.view({columns.size(0)*columns.size(1), columns.size(2)});
  }

  weight = weight.view({weight.size(0) * weight.size(1), weight.size(2),
                        weight.size(3), weight.size(4),weight.size(5)});
  output = output.view({output.size(0), output.size(1) * output.size(2),
                        output.size(3), output.size(4),output.size(5),output.size(6)});
  output = output.view({batch / step, channels_out, step, height_out, width_out,length_out});
  output.transpose_(1, 2);
  output = output.contiguous().view({batch , channels_out, height_out, width_out,length_out});
  if (with_bias)
    output += bias.view({1, bias.size(0), 1, 1, 1});
  input=input.view({batch,channels,height,width,length});
  offset=offset.view({batch,deformable_group * 3 *kernel_h*kernel_w*kernel_l,height_out,width_out,length_out});
  return 0;
}


template <typename scalar_t>
__global__ void deform_conv3d_gradient_gpu_kernel(
		const int n,const scalar_t *grad_col, const scalar_t *data_input,
		const scalar_t *data_offset,scalar_t * columns,
        const int channels_input,
        const int height_input, const int width_input, const int length_input,
        const int kernel_h, const int kernel_w, const int kernel_l,
        const int pad_h, const int pad_w, const int pad_l,
        const int stride_h, const int stride_w, const int stride_l,
        const int dilation_h, const int dilation_w, const int dilation_l,
        const int channel_per_deformable_group,const int step,
        const int offset_channels, const int deformable_group,
        const int height_col, const int width_col, const int length_col,
        scalar_t * grad_input,scalar_t *grad_offset){
// columns = {channels * kernel_h * kernel_w * kernel_l,step*height_out * width_out * length_out}
// grad_columns = {channels * kernel_h * kernel_w * kernel_l,step*height_out * width_out * length_out}
// grad_output = {step,channels_out,height_out,width_out , length_out}
// input = {step, channels, height, width, length}
// offset = {step,deformable_group * 3 * kernel_h * kernel_w * kernel_l,height_out,width_out,length_out}
// grad_offset = {step,deformable_group * 3 * kernel_h * kernel_w * kernel_l,height_out,width_out,length_out}
  CUDA_KERNEL_LOOP(index, n)//channels*kernel_h * kernel_w * kernel_l* step * height_col * width_col * length_col;
  {
	int single_col_len=length_col * width_col * height_col;
	int f = (index /step/ single_col_len )%(kernel_h * kernel_w * kernel_l);
	int i=(f / kernel_l / kernel_w) % kernel_h;
	int j=(f / kernel_l) %kernel_w;
	int k=f % kernel_l;

	int bpos=(index%(step*single_col_len))/(single_col_len);
	int lpos_col = (index % (single_col_len)) % length_col;
	int wpos_col = ((index % (single_col_len)) / length_col) % width_col;
    int hpos_col = ((index % (single_col_len)) / length_col / width_col) % height_col;
    int cpos_col = (index / step / single_col_len);
    int cpos_in=cpos_col/kernel_h/kernel_w/kernel_l;
    int offset_group_index=cpos_in/(channels_input/deformable_group);
    // printf("index %d cpos_col %d hpos_col %d wpos_col %d \n",index,cpos_col,hpos_col,wpos_col);
    int offset_base_ptr=bpos*(deformable_group * 3 * kernel_h * kernel_w * kernel_l*single_col_len)
    	    				+offset_group_index*channel_per_deformable_group*single_col_len
    	    				+hpos_col*width_col*length_col+wpos_col*length_col+lpos_col;
    int offset_h_ptr=offset_base_ptr+3*f*single_col_len;
    int offset_w_ptr=offset_base_ptr+(3*f+1)*single_col_len;
    int offset_l_ptr=offset_base_ptr+(3*f+2)*single_col_len;

    scalar_t offset_h=data_offset[offset_h_ptr];
    scalar_t offset_w=data_offset[offset_w_ptr];
    scalar_t offset_l=data_offset[offset_l_ptr];

    int hpos_in = hpos_col * stride_h  -pad_h + (i) * dilation_h;
    int wpos_in = wpos_col * stride_w - pad_w + (j) * dilation_w;
    int lpos_in = lpos_col * stride_l - pad_l + (k) * dilation_l;

    auto real_offset_h=hpos_in+offset_h;
    auto real_offset_w=wpos_in+offset_w;
    auto real_offset_l=lpos_in+offset_l;

    int h_low = floor(real_offset_h);
    int w_low = floor(real_offset_w);
    int l_low = floor(real_offset_l);

    int h_high = h_low + 1;
    int w_high = w_low + 1;
    int l_high = l_low + 1;

    scalar_t dh = real_offset_h - h_low;
    scalar_t dw = real_offset_w - w_low;
    scalar_t dl = real_offset_l - l_low;

    scalar_t w1 = (1-dh) *(1- dw)*(1-dl), w2 =(1- dh) *(1- dw)*dl, w3 = (1-dh)*dw*(1-dl), w4 = (1-dh) * dw*dl;
    scalar_t w5 = dh *(1- dw)*(1-dl), w6 =dh*(1- dw)*dl, w7 = dh*dw*(1-dl), w8 = dh*dw*dl;
    auto dval=grad_col[index];

    int data_input_base_ptr=(bpos*channels_input+cpos_in)*height_input*width_input*length_input;
    int grad_input_base_ptr=(bpos*channels_input+cpos_in)*height_input*width_input*length_input;

    bool h_low_flag=h_low  >= 0 && h_low <= height_input -1;
    bool w_low_flag=w_low >= 0  && w_low <= width_input - 1;
    bool l_low_flag=l_low >= 0  && l_low <= length_input -1;
    bool h_high_flag=h_high >= 0 && h_high <= height_input -1 &&  abs(dh)>EPS;
    bool w_high_flag=w_high >= 0 && w_high <= width_input - 1 &&  abs(dw)>EPS;
    bool l_high_flag=l_high >= 0 && l_high <= length_input -1 &&  abs(dl)>EPS;

    scalar_t v1 = static_cast<scalar_t>(0);
    if (h_low_flag  && w_low_flag  && l_low_flag ){
      v1 = data_input[data_input_base_ptr +h_low * width_input*length_input + w_low* length_input+l_low];
      atomicAdd(grad_input+grad_input_base_ptr +h_low * width_input*length_input + w_low* length_input+l_low,w1*dval);
    }
    scalar_t v2 = static_cast<scalar_t>(0);
    if (h_low_flag && w_low_flag  && l_high_flag ){
      v2 = data_input[data_input_base_ptr +h_low * width_input*length_input + w_low* length_input+l_high];
      atomicAdd(grad_input+grad_input_base_ptr +h_low * width_input*length_input + w_low* length_input+l_high,w2*dval);
    }
    scalar_t v3 = static_cast<scalar_t>(0);
    if (h_low_flag && w_high_flag  && l_low_flag ){
      v3 = data_input[data_input_base_ptr +h_low * width_input*length_input + w_high* length_input+l_low];
      atomicAdd(grad_input+grad_input_base_ptr +h_low * width_input*length_input + w_high* length_input+l_low,w3*dval);
    }
    scalar_t v4 = static_cast<scalar_t>(0);
    if (h_low_flag && w_high_flag  && l_high_flag ){
      v4 = data_input[data_input_base_ptr +h_low * width_input*length_input + w_high* length_input+l_high];
      atomicAdd(grad_input+grad_input_base_ptr +h_low * width_input*length_input + w_high* length_input+l_high,w4*dval);
    }
    scalar_t v5 = static_cast<scalar_t>(0);
    if (h_high_flag  && w_low_flag  && l_low_flag ){
      v5 = data_input[data_input_base_ptr +h_high * width_input*length_input + w_low* length_input+l_low];
      atomicAdd(grad_input+grad_input_base_ptr +h_high * width_input*length_input + w_low* length_input+l_low,w5*dval);
    }
    scalar_t v6 = static_cast<scalar_t>(0);
    if (h_high_flag  && w_low_flag  && l_high_flag ){
      v6 = data_input[data_input_base_ptr +h_high * width_input*length_input + w_low* length_input+l_high];
      atomicAdd(grad_input+grad_input_base_ptr +h_high * width_input*length_input + w_low* length_input+l_high,w6*dval);
    }
    scalar_t v7 = static_cast<scalar_t>(0);
    if (h_high_flag  && w_high_flag  && l_low_flag ){
      v7 = data_input[data_input_base_ptr +h_high * width_input*length_input + w_high* length_input+l_low];
      atomicAdd(grad_input+grad_input_base_ptr +h_high * width_input*length_input + w_high* length_input+l_low,w7*dval);
    }
    scalar_t v8 = static_cast<scalar_t>(0);
    if (h_high_flag  && w_high_flag  && l_high_flag ){
      v8 = data_input[data_input_base_ptr +h_high * width_input*length_input + w_high* length_input+l_high];
      atomicAdd(grad_input+grad_input_base_ptr +h_high * width_input*length_input + w_high* length_input+l_high,w8*dval);
    }
    atomicAdd(grad_offset + offset_h_ptr,
    		(-1*(1-dw)*(1-dl)*v1-1*(1-dw)*dl*v2-1*dw*(1-dl)*v3-1*dw*dl*v4+(1-dw)*(1-dl)*v5+(1-dw)*dl*v6+dw*(1-dl)*v7+dw*dl*v8)*dval);
    atomicAdd(grad_offset + offset_w_ptr,
    		(-1*(1-dh)*(1-dl)*v1-1*(1-dh)*dl*v2+(1-dh)*(1-dl)*v3+(1-dh)*dl*v4-1*dh*(1-dl)*v5-1*dh*dl*v6+dh*(1-dl)*v7+dh*dl*v8)*dval);
    atomicAdd(grad_offset + offset_l_ptr,
    		(-1*(1-dh)*(1-dw)*v1+(1-dh)*(1-dw)*v2-1*(1-dh)*dw*v3+(1-dh)*dw*v4-1*dh*(1-dw)*v5+dh*(1-dw)*v6-1*dh*dw*v7+dh*dw*v8)*dval);
    scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4+w5 * v5 + w6 * v6 + w7 * v7 + w8 * v8);
    columns[index]=val;
  }
}

// gradient offset mask input
void deform_conv3d_gradient_cuda(
    at::Tensor grad_col,at::Tensor data_input, at::Tensor data_offset,at::Tensor columns,
    const int channels, const int height_input, const int width_input, const int length_input,
    const int height_col, const int width_col, const int length_col,
    const int kernel_h, const int kernel_w, const int kernel_l,
    const int pad_h, const int pad_w, const int pad_l,
    const int stride_h, const int stride_w, const int stride_l,
    const int dilation_h, const int dilation_w, const int dilation_l,
    const int step,const int deformable_group,
    at::Tensor grad_input, at::Tensor grad_offset)
{
  const int num_kernels =channels*height_col * width_col * length_col * kernel_h * kernel_w * kernel_l *step;
  const int channel_per_deformable_group =3 * kernel_h * kernel_w * kernel_l;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_col.scalar_type(), "deform_conv3d_gradient_gpu_kernel", ([&] {
        const scalar_t *grad_col_ = grad_col.data<scalar_t>();
        const scalar_t *data_input_ = data_input.data<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data<scalar_t>();
        scalar_t *columns_ = columns.data<scalar_t>();
        scalar_t *grad_input_ = grad_input.data<scalar_t>();
        scalar_t *grad_offset_ = grad_offset.data<scalar_t>();

        deform_conv3d_gradient_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, grad_col_, data_input_, data_offset_,columns_,
            channels, height_input, width_input, length_input,
            kernel_h, kernel_w, kernel_l,
            pad_h, pad_w, pad_l,
            stride_h, stride_w, stride_l,
            dilation_h, dilation_w, dilation_l,
            channel_per_deformable_group, step,
            channel_per_deformable_group * deformable_group,
            deformable_group, height_col, width_col, length_col,
            grad_input_,grad_offset_);
      }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in deform_conv3d_gradient_cuda: %s\n", cudaGetErrorString(err));
  }
}

int deform_conv3d_backward_cuda(
    at::Tensor input, at::Tensor weight, at::Tensor bias,at::Tensor offset,
    at::Tensor grad_input, at::Tensor grad_weight, at::Tensor grad_bias,
    at::Tensor grad_offset, at::Tensor grad_output,
    const int kernel_h,const int kernel_w,const int kernel_l,
    const int stride_h,const int stride_w,const int stride_l,
    const int pad_h,const int pad_w,const int pad_l,
    const int dilation_h,const int dilation_w,const int dilation_l,
    const int group, int deformable_group,const int in_step,const bool with_bias) {
  TORCH_CHECK(input.is_contiguous(), "input tensor has to be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight tensor has to be contiguous");
  TORCH_CHECK(bias.is_contiguous(), "bias tensor has to be contiguous");
  TORCH_CHECK(offset.is_contiguous(), "offset tensor has to be contiguous");

  TORCH_CHECK(grad_input.is_contiguous(), "grad_input tensor has to be contiguous");
  TORCH_CHECK(grad_weight.is_contiguous(), "grad_weight tensor has to be contiguous");
  TORCH_CHECK(grad_bias.is_contiguous(), "grad_bias tensor has to be contiguous");
  TORCH_CHECK(grad_offset.is_contiguous(), "grad_offset tensor has to be contiguous");
  TORCH_CHECK(grad_output.is_contiguous(), "grad_output tensor has to be contiguous");

  const int batch = input.size(0);
  const int channels = input.size(1);
  const int height = input.size(2);
  const int width = input.size(3);
  const int length = input.size(4);

  const int channels_out = weight.size(0);
  const int channels_kernel = weight.size(1);
  const int kernel_h_ = weight.size(2);
  const int kernel_w_ = weight.size(3);
  const int kernel_l_ = weight.size(4);
  if (kernel_h_ != kernel_h || kernel_w_ != kernel_w || kernel_l_ != kernel_l)
    AT_ERROR("Input shape and kernel shape wont match: (%d x %d x %d vs %d x %d x %d).",
             kernel_h_, kernel_w_, kernel_l_, kernel_h, kernel_w, kernel_l);
  if (channels != channels_kernel * group)
    AT_ERROR("Input shape and kernel channels wont match: (%d vs %d).",
             channels, channels_kernel * group);

  const int height_out =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_out =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int length_out =
      (length + 2 * pad_l - (dilation_l * (kernel_l - 1) + 1)) / stride_l + 1;

  const int step=GET_STEP(batch,in_step);

  at::Tensor ones = at::ones({step,height_out, width_out, length_out}, input.options());
  at::Tensor columns = at::zeros({channels * kernel_h * kernel_w * kernel_l,
	  	  	  	  	  	  	  	  step*height_out * width_out * length_out},input.options());
  at::Tensor grad_columns=at::zeros({channels * kernel_h * kernel_w * kernel_l,
	  	  	  	  	  	  	  	  step*height_out * width_out * length_out},input.options());

  grad_output=grad_output.view({batch/step,step,channels_out,height_out,width_out,length_out});
  grad_output.transpose_(1, 2);
  grad_output =grad_output.view({grad_output.size(0), group, grad_output.size(1) / group,
                        grad_output.size(2), grad_output.size(3),grad_output.size(4),grad_output.size(5)});

  input=input.view({batch/step,step,channels,height,width,length});
  grad_input = grad_input.view({batch/step,step, channels, height, width,length});
  offset=offset.view({batch/step,step,
	  deformable_group * 3 * kernel_h * kernel_w * kernel_l,height_out,width_out,length_out});
  grad_offset=grad_offset.view({batch/step,step,
	  deformable_group * 3 * kernel_h * kernel_w * kernel_l,height_out,width_out,length_out});

  for (int b = 0; b < batch/step; b++) {
    // divide int group
	grad_columns = grad_columns.view({group, grad_columns.size(0) / group, grad_columns.size(1)});
    weight = weight.view({group, weight.size(0) / group, weight.size(1),weight.size(2), weight.size(3), weight.size(4)});
    for (int g = 0; g < group; g++) {
    	grad_columns[g].addmm_(weight[g].flatten(1).transpose(0, 1),
                        grad_output[b][g].flatten(1), 0.0f, 1.0f);
    }
    grad_columns = grad_columns.view({grad_columns.size(0) * grad_columns.size(1), grad_columns.size(2)});
    weight = weight.view({weight.size(0) * weight.size(1), weight.size(2),
                          weight.size(3), weight.size(4), weight.size(5)});
    //print_tensor_size("grad_columns size",grad_columns);
    //print_tensor_size("grad_mask[b] size",grad_mask[b]);
    columns.fill_(0);
    deform_conv3d_gradient_cuda(
    		grad_columns, input[b], offset[b], columns,
    		channels, height, width, length,
    		height_out, width_out, length_out,
    		kernel_h, kernel_w, kernel_l,
    		pad_h, pad_w, pad_l,
    		stride_h, stride_w, stride_l,
    		dilation_h, dilation_w, dilation_l,
    		step,deformable_group,
    		grad_input[b],grad_offset[b]);
    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    grad_weight = grad_weight.view({group, grad_weight.size(0) / group,
                                    grad_weight.size(1), grad_weight.size(2),
                                    grad_weight.size(3),grad_weight.size(4)});
    if (with_bias)
      grad_bias = grad_bias.view({group, grad_bias.size(0) / group});
    for (int g = 0; g < group; g++) {
      grad_weight[g] =grad_weight[g].flatten(1)
    				.addmm_(grad_output[b][g].flatten(1),columns[g].transpose(0, 1),1.0f,1.0f)
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
                                    grad_weight.size(4), grad_weight.size(5)});
    if (with_bias)
      grad_bias = grad_bias.view({grad_bias.size(0) * grad_bias.size(1)});
  }
  grad_output = grad_output.view({grad_output.size(0) ,grad_output.size(1)*grad_output.size(2),
	  	  	  	  	  	  	  	  grad_output.size(3),grad_output.size(4),
	  	  	  	  	  	  	  	  grad_output.size(5),grad_output.size(6)});
  //grad_output=grad_output.view({batch/step,channels_kernel,step,height_out,width_out,length_out});
  grad_output.transpose_(1, 2);
  grad_output =grad_output.view({batch,channels_out,height_out,width_out,length_out});

  input=input.view({batch,channels,height,width,length});
  grad_input = grad_input.view({batch, channels, height, width,length});

  offset=offset.view({batch,deformable_group * 3 * kernel_h * kernel_w *kernel_l,
	  	  	  	  	  height_out,width_out,length_out});
  grad_offset=grad_offset.view({batch,deformable_group * 3 * kernel_h * kernel_w *kernel_l,
  	  	  	  	  	  height_out,width_out,length_out});
  return 0;
}








