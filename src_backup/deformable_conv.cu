#include "config.h"

void shape_check(at::Tensor input, at::Tensor offset, at::Tensor *gradOutput,
                 at::Tensor weight, int kH, int kW, int dH, int dW, int padH,
                 int padW, int dilationH, int dilationW, int group,
                 int deformable_group) {
  TORCH_CHECK(weight.ndimension() == 4,"4D weight tensor (nOutputPlane,nInputPlane,kH,kW) expected, but got: %s",weight.ndimension());

  TORCH_CHECK(weight.is_contiguous(), "weight tensor has to be contiguous");

  TORCH_CHECK(kW > 0 && kH > 0,"kernel size should be greater than zero, but got kH: %d kW: %d", kH,kW);

  TORCH_CHECK((weight.size(2) == kH && weight.size(3) == kW),"kernel size should be consistent with weight, ",
         "but got kH: %d kW: %d weight.size(2): %d, weight.size(3): %d", kH,kW, weight.size(2), weight.size(3));

  TORCH_CHECK(dW > 0 && dH > 0,"stride should be greater than zero, but got dH: %d dW: %d", dH, dW);

  TORCH_CHECK(dilationW > 0 && dilationH > 0,"dilation should be greater than 0, but got dilationH: %d dilationW: %d",dilationH, dilationW);

  int ndim = input.ndimension();
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  AT_CHECK(ndim == 3 || ndim == 4, "3D or 4D input tensor expected but got: %s",ndim);

  long nInputPlane = weight.size(1) * group;
  long inputHeight = input.size(dimh);
  long inputWidth = input.size(dimw);
  long nOutputPlane = weight.size(0);
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;

  TORCH_CHECK(nInputPlane % deformable_group == 0,"input channels must divide deformable group size");

  if (outputWidth < 1 || outputHeight < 1)
    AT_ERROR(
        "Given input size: (%ld x %ld x %ld). "
        "Calculated output size: (%ld x %ld x %ld). Output size is too small",
        nInputPlane, inputHeight, inputWidth, nOutputPlane, outputHeight,
        outputWidth);

  TORCH_CHECK(input.size(1) == nInputPlane,"invalid number of input planes, expected: %d, but got: %d",nInputPlane, input.size(1));

  TORCH_CHECK((inputHeight >= kH && inputWidth >= kW),"input image is smaller than kernel");

  TORCH_CHECK((offset.size(2) == outputHeight && offset.size(3) == outputWidth),
           "invalid spatial size of offset, expected height: %d width: %d, but got height: %d width: %d",outputHeight, outputWidth, offset.size(2), offset.size(3));

  TORCH_CHECK((offset.size(1) == deformable_group * 2 * kH * kW),"invalid number of channels of offset");

  if (gradOutput != NULL) {
	  TORCH_CHECK(gradOutput->size(dimf) == nOutputPlane,"invalid number of gradOutput planes, expected: %d, but got: %d",nOutputPlane, gradOutput->size(dimf));

	  TORCH_CHECK((gradOutput->size(dimh) == outputHeight &&gradOutput->size(dimw) == outputWidth),
             "invalid size of gradOutput, expected height: %d width: %d , but "
             "got height: %d width: %d",
             outputHeight, outputWidth, gradOutput->size(dimh),
             gradOutput->size(dimw));
  }
}

template <typename scalar_t>
__device__ scalar_t deform_conv2d_im2col_bilinear(const scalar_t *bottom_data, const int data_width,
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
__global__ void deform_conv2d_im2col_gpu_kernel(const int n, const scalar_t *data_im, const scalar_t *data_offset,
                                             const int height, const int width, const int kernel_h, const int kernel_w,
                                             const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                                             const int dilation_h, const int dilation_w, const int channel_per_deformable_group,
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
    //const scalar_t* data_im_ptr = data_im + ((b_col * num_channels + c_im) * height + h_in) * width + w_in;
    const scalar_t *data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width;
    const scalar_t *data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i)
    {
      for (int j = 0; j < kernel_w; ++j)
      {
        const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
        const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
        const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
        scalar_t val = static_cast<scalar_t>(0);
        const scalar_t h_im = h_in + i * dilation_h + offset_h;
        const scalar_t w_im = w_in + j * dilation_w + offset_w;
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width)
        {
          //const scalar_t map_h = i * dilation_h + offset_h;
          //const scalar_t map_w = j * dilation_w + offset_w;
          //const int cur_height = height - h_in;
          //const int cur_width = width - w_in;
          //val = deformable_im2col_bilinear(data_im_ptr, width, cur_height, cur_width, map_h, map_w);
          val = deform_conv2d_im2col_bilinear(data_im_ptr, width, height, width, h_im, w_im);
        }
        *data_col_ptr = val;
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
}

void deform_conv2d_im2col(
    const at::Tensor data_im, const at::Tensor data_offset, const int channels,
    const int height, const int width, const int ksize_h, const int ksize_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int parallel_imgs,
    const int deformable_group, at::Tensor data_col)
{
  // num_axes should be smaller than block size
  // todo: check parallel_imgs is correctly passed in
  int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col * parallel_imgs;
  int channel_per_deformable_group = channels / deformable_group;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_im.scalar_type(), "deform_conv2d_im2col_gpu_kernel", ([&] {
        const scalar_t *data_im_ = data_im.data_ptr<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
        scalar_t *data_col_ = data_col.data_ptr<scalar_t>();

        deform_conv2d_im2col_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, data_im_, data_offset_, height, width, ksize_h, ksize_w,
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
            channel_per_deformable_group, parallel_imgs, channels, deformable_group,
            height_col, width_col, data_col_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in deform_conv2d_im2col: %s\n", cudaGetErrorString(err));
  }
}

int deform_conv2d_forward_cuda(at::Tensor input, at::Tensor weight,
                             at::Tensor offset, at::Tensor output,
                             int kH, int kW, int dH, int dW, int padH, int padW,
                             int dilationH, int dilationW, int group,
                             int deformable_group, int im2col_step) {
  // todo: add im2col_step as input
  // todo: add new output buffer and transpose it to output (or directly
  // transpose output) todo: possibly change data indexing because of
  // parallel_imgs

  shape_check(input, offset, NULL, weight, kH, kW, dH, dW, padH, padW,
              dilationH, dilationW, group, deformable_group);

  input = input.contiguous();
  offset = offset.contiguous();
  weight = weight.contiguous();

  int batch = 1;
  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input.unsqueeze_(0);
    offset.unsqueeze_(0);
  }

  // todo: assert batchsize dividable by im2col_step

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = weight.size(0);

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  TORCH_CHECK((offset.size(0) == batchSize), "invalid batch size of offset");

  output = output.view({batchSize / im2col_step, im2col_step, nOutputPlane,
                        outputHeight, outputWidth});

  input = input.view({batchSize / im2col_step, im2col_step, nInputPlane,
                      inputHeight, inputWidth});
  offset =
      offset.view({batchSize / im2col_step, im2col_step,
                   deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  at::Tensor output_buffer =
      at::zeros({batchSize / im2col_step, nOutputPlane,
                 im2col_step * outputHeight, outputWidth},
                output.options());

  output_buffer = output_buffer.view(
      {output_buffer.size(0), group, output_buffer.size(1) / group,
       output_buffer.size(2), output_buffer.size(3)});
  weight = weight.view({group, weight.size(0) / group, weight.size(1),
                        weight.size(2), weight.size(3)});
  at::Tensor columns = at::zeros(
	      {nInputPlane * kW * kH, im2col_step * outputHeight * outputWidth},
	      input.options());
  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    columns.fill_(0);
    deform_conv2d_im2col(input[elt], offset[elt], nInputPlane, inputHeight,
                      inputWidth, kH, kW, padH, padW, dH, dW, dilationH,
                      dilationW, im2col_step, deformable_group, columns);
    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    for (int g = 0; g < group; g++) {
      output_buffer[elt][g] = output_buffer[elt][g]
                                  .flatten(1)
                                  .addmm_(weight[g].flatten(1), columns[g],1.0f,1.0f)
                                  .view_as(output_buffer[elt][g]);
    }
  }

  output_buffer = output_buffer.view(
      {output_buffer.size(0), output_buffer.size(1) * output_buffer.size(2),
       output_buffer.size(3), output_buffer.size(4)});

  output_buffer = output_buffer.view({batchSize / im2col_step, nOutputPlane,
                                      im2col_step, outputHeight, outputWidth});
  output_buffer.transpose_(1, 2);
  output.copy_(output_buffer);
  output = output.view({batchSize, nOutputPlane, outputHeight, outputWidth});

  input = input.view({batchSize, nInputPlane, inputHeight, inputWidth});
  offset = offset.view(
      {batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  if (batch == 0) {
    output = output.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
    offset = offset.view({offset.size(1), offset.size(2), offset.size(3)});
  }

  return 0;
}

template <typename scalar_t>
__device__ scalar_t get_gradient_weight(scalar_t argmax_h, scalar_t argmax_w,
                                        const int h, const int w, const int height, const int width)
{

  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width)
  {
    //empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  scalar_t weight = 0;
  if (h == argmax_h_low && w == argmax_w_low)
    weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high)
    weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (h == argmax_h_high && w == argmax_w_low)
    weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high)
    weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
  return weight;
}

template <typename scalar_t>
__device__ scalar_t get_coordinate_weight(scalar_t argmax_h, scalar_t argmax_w,
                                          const int height, const int width, const scalar_t *im_data,
                                          const int data_width, const int bp_dir)
{

  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width)
  {
    //empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  scalar_t weight = 0;

  if (bp_dir == 0)
  {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += -1 * (argmax_w - argmax_w_low) * im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_w - argmax_w_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  }
  else if (bp_dir == 1)
  {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += -1 * (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}

template <typename scalar_t>
__global__ void deform_conv2d_col2im_gpu_kernel(
    const int n, const scalar_t *data_col, const scalar_t *data_offset,
    const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group,
    const int batch_size, const int deformable_group,
    const int height_col, const int width_col,
    scalar_t *grad_im)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    const int j = (index / width_col / height_col / batch_size) % kernel_w;
    const int i = (index / width_col / height_col / batch_size / kernel_w) % kernel_h;
    const int c = index / width_col / height_col / batch_size / kernel_w / kernel_h;
    // compute the start and end of the output

    const int deformable_group_index = c / channel_per_deformable_group;

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int b = (index / width_col / height_col) % batch_size;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    const scalar_t *data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) *
                                                        2 * kernel_h * kernel_w * height_col * width_col;
    const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
    const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
    const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
    const scalar_t cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const scalar_t cur_inv_w_data = w_in + j * dilation_w + offset_w;

    const scalar_t cur_top_grad = data_col[index];
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;
    for (int dy = -2; dy <= 2; dy++)
    {
      for (int dx = -2; dx <= 2; dx++)
      {
        if (cur_h + dy >= 0 && cur_h + dy < height &&
            cur_w + dx >= 0 && cur_w + dx < width &&
            abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1)
        {
          int cur_bottom_grad_pos = ((b * channels + c) * height + cur_h + dy) * width + cur_w + dx;
          scalar_t weight = get_gradient_weight(cur_inv_h_data, cur_inv_w_data, cur_h + dy, cur_w + dx, height, width);
          atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
        }
      }
    }
  }
}

void deform_conv2d_col2im(
    const at::Tensor data_col, const at::Tensor data_offset, const int channels,
    const int height, const int width, const int ksize_h,
    const int ksize_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int parallel_imgs, const int deformable_group,
    at::Tensor grad_im)
{

  // todo: make sure parallel_imgs is passed in correctly
  int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * ksize_h * ksize_w * height_col * width_col * parallel_imgs;
  int channel_per_deformable_group = channels / deformable_group;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.scalar_type(), "deform_conv2d_col2im_gpu_kernel", ([&] {
        const scalar_t *data_col_ = data_col.data_ptr<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
        scalar_t *grad_im_ = grad_im.data_ptr<scalar_t>();

        deform_conv2d_col2im_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, data_col_, data_offset_, channels, height, width, ksize_h,
            ksize_w, pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, channel_per_deformable_group,
            parallel_imgs, deformable_group, height_col, width_col, grad_im_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in deform_conv2d_col2im: %s\n", cudaGetErrorString(err));
  }
}

template <typename scalar_t>
__global__ void deform_conv2d_col2im_coord_gpu_kernel(
		const int n, const scalar_t *data_col,
        const scalar_t *data_im, const scalar_t *data_offset,
        const int channels, const int height, const int width,
        const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        const int channel_per_deformable_group,
        const int batch_size, const int offset_channels, const int deformable_group,
        const int height_col, const int width_col, scalar_t *grad_offset)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    scalar_t val = 0;
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int c = (index / width_col / height_col) % offset_channels;
    int b = (index / width_col / height_col) / offset_channels;
    // compute the start and end of the output

    const int deformable_group_index = c / (2 * kernel_h * kernel_w);
    const int col_step = kernel_h * kernel_w;
    int cnt = 0;
    const scalar_t *data_col_ptr = data_col + deformable_group_index * channel_per_deformable_group *
                                                  batch_size * width_col * height_col;
    const scalar_t *data_im_ptr = data_im + (b * deformable_group + deformable_group_index) *
                                                channel_per_deformable_group / kernel_h / kernel_w * height * width;
    const scalar_t *data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) * 2 *
                                                        kernel_h * kernel_w * height_col * width_col;

    const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

    for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group; col_c += col_step)
    {
      const int col_pos = (((col_c * batch_size + b) * height_col) + h) * width_col + w;
      const int bp_dir = offset_c % 2;

      int j = (col_pos / width_col / height_col / batch_size) % kernel_w;
      int i = (col_pos / width_col / height_col / batch_size / kernel_w) % kernel_h;
      int w_out = col_pos % width_col;
      int h_out = (col_pos / width_col) % height_col;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      const int data_offset_h_ptr = (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
      const int data_offset_w_ptr = (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out);
      const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
      const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
      scalar_t inv_h = h_in + i * dilation_h + offset_h;
      scalar_t inv_w = w_in + j * dilation_w + offset_w;
      if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width)
      {
        inv_h = inv_w = -2;
      }
      const scalar_t weight = get_coordinate_weight(
          inv_h, inv_w,height, width, data_im_ptr + cnt * height * width, width, bp_dir);
      val += weight * data_col_ptr[col_pos];
      cnt += 1;
    }

    grad_offset[index] = val;
  }
}

void deform_conv2d_col2im_coord(
    const at::Tensor data_col, const at::Tensor data_im, const at::Tensor data_offset,
    const int channels, const int height, const int width, const int ksize_h,
    const int ksize_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int parallel_imgs, const int deformable_group, at::Tensor grad_offset)
{

  int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = height_col * width_col * 2 * ksize_h * ksize_w * deformable_group * parallel_imgs;
  int channel_per_deformable_group = channels * ksize_h * ksize_w / deformable_group;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.scalar_type(), "deform_conv2d_col2im_coord_gpu_kernel", ([&] {
        const scalar_t *data_col_ = data_col.data_ptr<scalar_t>();
        const scalar_t *data_im_ = data_im.data_ptr<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
        scalar_t *grad_offset_ = grad_offset.data_ptr<scalar_t>();

        deform_conv2d_col2im_coord_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, data_col_, data_im_, data_offset_, channels, height, width,
            ksize_h, ksize_w, pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, channel_per_deformable_group,
            parallel_imgs, 2 * ksize_h * ksize_w * deformable_group, deformable_group,
            height_col, width_col, grad_offset_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in deform_conv2d_col2im_coord: %s\n", cudaGetErrorString(err));
  }
}

int deform_conv2d_backward_input_cuda(
		at::Tensor input, at::Tensor offset,
        at::Tensor gradOutput, at::Tensor gradInput,
        at::Tensor gradOffset, at::Tensor weight,
        int kH, int kW, int dH,int dW, int padH, int padW, int dilationH,int dilationW,
        int group,int deformable_group, int im2col_step) {
  shape_check(input, offset, &gradOutput, weight, kH, kW, dH, dW, padH, padW,
              dilationH, dilationW, group, deformable_group);

  input = input.contiguous();
  offset = offset.contiguous();
  gradOutput = gradOutput.contiguous();
  weight = weight.contiguous();

  int batch = 1;

  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input = input.view({1, input.size(0), input.size(1), input.size(2)});
    offset = offset.view({1, offset.size(0), offset.size(1), offset.size(2)});
    gradOutput = gradOutput.view(
        {1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2)});
  }

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = weight.size(0);

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  TORCH_CHECK((offset.size(0) == batchSize), 3, "invalid batch size of offset");
  gradInput = gradInput.view({batchSize, nInputPlane, inputHeight, inputWidth});
  at::Tensor columns = at::zeros(
      {nInputPlane * kW * kH, im2col_step * outputHeight * outputWidth},
      input.options());

  // change order of grad output
  gradOutput = gradOutput.view({batchSize / im2col_step, im2col_step,
                                nOutputPlane, outputHeight, outputWidth});
  gradOutput.transpose_(1, 2);

  gradInput = gradInput.view({batchSize / im2col_step, im2col_step, nInputPlane,
                              inputHeight, inputWidth});
  input = input.view({batchSize / im2col_step, im2col_step, nInputPlane,
                      inputHeight, inputWidth});
  gradOffset = gradOffset.view({batchSize / im2col_step, im2col_step,
                                deformable_group * 2 * kH * kW, outputHeight,
                                outputWidth});
  offset =
      offset.view({batchSize / im2col_step, im2col_step,
                   deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    // divide into groups
    columns.fill_(0);
    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    weight = weight.view({group, weight.size(0) / group, weight.size(1),
                          weight.size(2), weight.size(3)});
    gradOutput = gradOutput.view(
        {gradOutput.size(0), group, gradOutput.size(1) / group,
         gradOutput.size(2), gradOutput.size(3), gradOutput.size(4)});

    for (int g = 0; g < group; g++) {
      columns[g] = columns[g].addmm_(weight[g].flatten(1).transpose(0, 1),
                                     gradOutput[elt][g].flatten(1), 0.0f, 1.0f);
    }

    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
    gradOutput = gradOutput.view(
        {gradOutput.size(0), gradOutput.size(1) * gradOutput.size(2),
         gradOutput.size(3), gradOutput.size(4), gradOutput.size(5)});

    deform_conv2d_col2im_coord(columns, input[elt], offset[elt], nInputPlane,
                            inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
                            dilationH, dilationW, im2col_step, deformable_group,
                            gradOffset[elt]);

    deform_conv2d_col2im(columns, offset[elt], nInputPlane, inputHeight,
                      inputWidth, kH, kW, padH, padW, dH, dW, dilationH,
                      dilationW, im2col_step, deformable_group, gradInput[elt]);
  }

  gradOutput.transpose_(1, 2);
  gradOutput =
      gradOutput.view({batchSize, nOutputPlane, outputHeight, outputWidth});

  gradInput = gradInput.view({batchSize, nInputPlane, inputHeight, inputWidth});
  input = input.view({batchSize, nInputPlane, inputHeight, inputWidth});
  gradOffset = gradOffset.view(
      {batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth});
  offset = offset.view(
      {batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  if (batch == 0) {
    gradOutput = gradOutput.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
    gradInput = gradInput.view({nInputPlane, inputHeight, inputWidth});
    offset = offset.view({offset.size(1), offset.size(2), offset.size(3)});
    gradOffset =
        gradOffset.view({offset.size(1), offset.size(2), offset.size(3)});
  }

  return 0;
}

int deform_conv2d_backward_parameters_cuda(
    at::Tensor input, at::Tensor offset, at::Tensor gradOutput,at::Tensor gradWeight,
    int kH, int kW, int dH, int dW,int padH, int padW, int dilationH, int dilationW,
    int group,int deformable_group, int im2col_step) {
  // todo: transpose and reshape outGrad
  // todo: add im2col_step as input

  shape_check(input, offset, &gradOutput, gradWeight, kH, kW, dH, dW, padH,
              padW, dilationH, dilationW, group, deformable_group);

  input = input.contiguous();
  offset = offset.contiguous();
  gradOutput = gradOutput.contiguous();

  int batch = 1;

  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input = input.view(
        at::IntList({1, input.size(0), input.size(1), input.size(2)}));
    gradOutput = gradOutput.view(
        {1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2)});
  }

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = gradWeight.size(0);

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  TORCH_CHECK((offset.size(0) == batchSize), "invalid batch size of offset");

  at::Tensor columns = at::zeros(
      {nInputPlane * kW * kH, im2col_step * outputHeight * outputWidth},
      input.options());

  gradOutput = gradOutput.view({batchSize / im2col_step, im2col_step,
                                nOutputPlane, outputHeight, outputWidth});
  gradOutput.transpose_(1, 2);

  at::Tensor gradOutputBuffer = at::zeros_like(gradOutput);
  gradOutputBuffer =
      gradOutputBuffer.view({batchSize / im2col_step, nOutputPlane, im2col_step,
                             outputHeight, outputWidth});
  gradOutputBuffer.copy_(gradOutput);
  gradOutputBuffer =
      gradOutputBuffer.view({batchSize / im2col_step, nOutputPlane,
                             im2col_step * outputHeight, outputWidth});

  gradOutput.transpose_(1, 2);
  gradOutput =
      gradOutput.view({batchSize, nOutputPlane, outputHeight, outputWidth});

  input = input.view({batchSize / im2col_step, im2col_step, nInputPlane,
                      inputHeight, inputWidth});
  offset = offset.view({batchSize / im2col_step, im2col_step,
                   deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    columns.fill_(0);
    deform_conv2d_im2col(input[elt], offset[elt], nInputPlane, inputHeight,
                      inputWidth, kH, kW, padH, padW, dH, dW, dilationH,
                      dilationW, im2col_step, deformable_group, columns);

    // divide into group
    gradOutputBuffer = gradOutputBuffer.view(
        {gradOutputBuffer.size(0), group, gradOutputBuffer.size(1) / group,
         gradOutputBuffer.size(2), gradOutputBuffer.size(3)});
    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    gradWeight = gradWeight.view({group, gradWeight.size(0) / group, gradWeight.size(1),
                         gradWeight.size(2), gradWeight.size(3)});

    for (int g = 0; g < group; g++) {
      gradWeight[g] = gradWeight[g]
                          .flatten(1)
                          .addmm_(gradOutputBuffer[elt][g].flatten(1),columns[g].transpose(1, 0), 1.0f, 1.0f)
                          .view_as(gradWeight[g]);
    }
    gradOutputBuffer = gradOutputBuffer.view(
        {gradOutputBuffer.size(0),
         gradOutputBuffer.size(1) * gradOutputBuffer.size(2),
         gradOutputBuffer.size(3), gradOutputBuffer.size(4)});
    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
    gradWeight = gradWeight.view({gradWeight.size(0) * gradWeight.size(1),
                                  gradWeight.size(2), gradWeight.size(3),
                                  gradWeight.size(4)});
  }

  input = input.view({batchSize, nInputPlane, inputHeight, inputWidth});
  offset = offset.view(
      {batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  if (batch == 0) {
    gradOutput = gradOutput.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
  }
  return 0;
}

template <typename scalar_t>
__global__ void deform_conv2d_gradient_gpu_kernel(
		const int n,const scalar_t *grad_col, const scalar_t *data_input, const scalar_t *data_offset,
		scalar_t * columns,
        const int channels_input, const int height_input, const int width_input,
        const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w, const int stride_h, const int stride_w,const int dilation_h, const int dilation_w,
        const int channel_per_deformable_group,
        const int offset_channels,
        const int deformable_group,
        const int height_col, const int width_col,
        scalar_t * grad_input,scalar_t *grad_offset)
{
  CUDA_KERNEL_LOOP(index, n)
  {
	int k = (index / width_col / height_col)%(kernel_h*kernel_w);
	int i=k/kernel_w;
	int j=k%kernel_w;
    int wpos_col = index % width_col;
    int hpos_col = (index / width_col) % height_col;
    int cpos_col = (index / width_col / height_col);
    int cpos_in=cpos_col/kernel_h/kernel_w;
    int offset_group_index=cpos_in/(channels_input/deformable_group);
    //printf("index %d cpos_col %d hpos_col %d wpos_col %d \n",index,cpos_col,hpos_col,wpos_col);
    int offset_h_ptr=offset_group_index*channel_per_deformable_group*height_col*width_col+2*k*height_col*width_col+hpos_col*width_col+wpos_col;
    int offset_w_ptr=offset_group_index*channel_per_deformable_group*height_col*width_col+(2*k+1)*height_col*width_col+hpos_col*width_col+wpos_col;
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
      v1 = data_input[cpos_in*height_input*width_input+h_low * width_input + w_low];
    scalar_t v2 = 0;
    if (h_low  >= 0 && h_low <= height_input - 1  && w_high >= 0  && w_high <= width_input - 1&&  abs(dw)>EPS )
      v2 = data_input[cpos_in*height_input*width_input+h_low * width_input + w_high];
    scalar_t v3 = 0;
    if (h_high >= 0 && h_high <= height_input - 1 && w_low  >= 0  && w_low <= width_input - 1 && abs(dh)>EPS)
      v3 = data_input[cpos_in*height_input*width_input+h_high * width_input + w_low];
    scalar_t v4 = 0;
    if (h_high >= 0 && h_high <= height_input - 1 && w_high >= 0  && w_high <= width_input - 1 && abs(dh)>EPS && abs(dw)>EPS )
      v4 = data_input[cpos_in*height_input*width_input+h_high * width_input + w_high];

    scalar_t w1 = (1-dh) *(1- dw), w2 =(1- dh) * dw, w3 = dh*(1- dw), w4 = dh * dw;
    scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    auto dval=grad_col[index];
    if (h_low >= 0  && h_low<= height_input - 1   && w_low >= 0    && w_low<= width_input - 1)
    	atomicAdd(grad_input + cpos_in*height_input*width_input+h_low * width_input + w_low, w1*dval);
    if (h_low >= 0  && h_low<= height_input - 1   && w_high>=0     && w_high <= width_input - 1)
    	atomicAdd(grad_input + cpos_in*height_input*width_input+h_low * width_input + w_high, w2*dval);
    if (h_high >= 0 && h_high <= height_input - 1 && w_low >= 0    && w_low<= width_input - 1)
    	atomicAdd(grad_input + cpos_in*height_input*width_input+h_high * width_input + w_low, w3*dval);
    if  (h_high >= 0 && h_high <= height_input - 1 && w_high>=0     && w_high <= width_input - 1)
    	atomicAdd(grad_input + cpos_in*height_input*width_input+h_high * width_input + w_high,w4*dval);
    atomicAdd(grad_offset + offset_h_ptr,(-1*(1-dw)*v1-1*dw*v2+(1-dw)*v3+dw*v4)*dval);
    atomicAdd(grad_offset + offset_w_ptr,(-1*(1-dh)*v1+(1-dh)*v2-1*dh*v3+dh*v4)*dval);
    columns[index]=val;
  }
}


// gradient offset mask input
void deform_conv2d_gradient_cuda(
    const at::Tensor grad_col,
    const at::Tensor data_input, const at::Tensor data_offset,at::Tensor columns,
    const int channels, const int height_input, const int width_input,
    const int height_col, const int width_col, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int deformable_group,
    at::Tensor grad_input, at::Tensor grad_offset)
{
  const int num_kernels =channels*height_col * width_col * kernel_h * kernel_w;
  const int channel_per_deformable_group =2 * kernel_h * kernel_w ;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_col.scalar_type(), "deform_conv2d_gradient_gpu_kernel", ([&] {
        const scalar_t *grad_col_ = grad_col.data<scalar_t>();
        const scalar_t *data_input_ = data_input.data<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data<scalar_t>();
        scalar_t *columns_ = columns.data<scalar_t>();
        scalar_t *grad_input_ = grad_input.data<scalar_t>();
        scalar_t *grad_offset_ = grad_offset.data<scalar_t>();

        deform_conv2d_gradient_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, grad_col_, data_input_, data_offset_,columns_,
            channels, height_input, width_input,
            kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
            channel_per_deformable_group, channel_per_deformable_group * deformable_group,
            deformable_group, height_col, width_col,
            grad_input_,grad_offset_);
      }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in deform_conv2d_gradient_cuda: %s\n", cudaGetErrorString(err));
  }
}


int deform_conv2d_backward_cuda_v2(
    at::Tensor input, at::Tensor weight, at::Tensor bias,at::Tensor offset,
    at::Tensor grad_input, at::Tensor grad_weight, at::Tensor grad_bias,
    at::Tensor grad_offset, at::Tensor grad_output,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
    int pad_w, int dilation_h, int dilation_w, int group, int deformable_group,
    const bool with_bias,int im2col_step) {

  AT_CHECK(input.is_contiguous(), "input tensor has to be contiguous");
  AT_CHECK(weight.is_contiguous(), "weight tensor has to be contiguous");

  const int batch = input.size(0);
  const int channels = input.size(1);
  const int height = input.size(2);
  const int width = input.size(3);

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

  at::Tensor ones = at::ones({height_out, width_out}, input.options());

  grad_input = grad_input.view({batch, channels, height, width});
  at::Tensor columns = at::zeros({channels * kernel_h * kernel_w, height_out * width_out},
                      input.options());

  at::Tensor grad_columns=at::zeros({channels * kernel_h * kernel_w, height_out * width_out},input.options());

  grad_output =grad_output.view({grad_output.size(0), group, grad_output.size(1) / group,
                        grad_output.size(2), grad_output.size(3)});

  for (int b = 0; b < batch; b++) {
    // divide int group
	grad_columns = grad_columns.view({group, grad_columns.size(0) / group, grad_columns.size(1)});
    weight = weight.view({group, weight.size(0) / group, weight.size(1),weight.size(2), weight.size(3)});
    for (int g = 0; g < group; g++) {
    	 grad_columns[g].addmm_(weight[g].flatten(1).transpose(0, 1),
                        grad_output[b][g].flatten(1), 0.0f, 1.0f);
    }
    grad_columns = grad_columns.view({grad_columns.size(0) * grad_columns.size(1), grad_columns.size(2)});
    weight = weight.view({weight.size(0) * weight.size(1), weight.size(2),
                          weight.size(3), weight.size(4)});
    columns.fill_(0);
    deform_conv2d_gradient_cuda(
    		grad_columns,input[b],offset[b],columns,
    		channels,height,width,height_out,width_out,
    		kernel_h,kernel_w,pad_h,pad_w,stride_h,stride_w,dilation_h, dilation_w,
    		deformable_group,grad_input[b],grad_offset[b]);
    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    grad_weight = grad_weight.view({group, grad_weight.size(0) / group,
                                    grad_weight.size(1), grad_weight.size(2),
                                    grad_weight.size(3)});
    if (with_bias)
      grad_bias = grad_bias.view({group, grad_bias.size(0) / group});

    for (int g = 0; g < group; g++) {
      grad_weight[g] =
          grad_weight[g].flatten(1).addmm_(grad_output[b][g].flatten(1), columns[g].transpose(0, 1),1.0f,1.0f)
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
  grad_output = grad_output.view({grad_output.size(0) * grad_output.size(1),
                                  grad_output.size(2), grad_output.size(3),
                                  grad_output.size(4)});
  return 0;
}




