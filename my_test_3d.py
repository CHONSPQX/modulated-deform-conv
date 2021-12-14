from modulated_deform_conv import *
import ipdb
print_tensor = lambda n, x: print(n, type(x), x.dtype, x.shape, x.min(), x.max())

batch_size=1
cpudata=torch.ones(batch_size,1,5,5,5, requires_grad=True)
# data=torch.ones(batch,1,5,5,device='cuda',requires_grad=True)
data=cpudata.cuda()

offset=torch.zeros(batch_size,81,5,5,5, device='cuda',requires_grad=True) # 3**3 kernel_size, *3 dims
mask=torch.ones(batch_size,27,5,5,5, device='cuda',requires_grad=True) # 3**3 kernel_size,
weight=torch.ones(1,1,3,3,3,device='cuda',requires_grad=True)
bias=torch.zeros(1,device='cuda',requires_grad=True)

print_tensor('input', data)
print_tensor('offset', offset)
print_tensor('mask', mask)
print_tensor('weight', weight)

stride=1
padding=1
dilation=1
groups=1
deformable_groups=1
in_step=64
'''
class DeformConv2dFunction(Function):
    @staticmethod
    def forward(ctx, input, offset, weight, bias=None, stride=1, padding=0, dilation=1,
                groups=1, deformable_groups=1 , in_step=64):
'''
# out=deform_conv3d(data,offset,weight,bias,stride,padding,dilation,groups,deformable_groups,in_step)
out=modulated_deform_conv3d(data,offset,mask,weight,bias,stride,padding,dilation,groups,deformable_groups,in_step)
print_tensor('out', out)

loss=out.sum()
print(loss)
# ipdb.set_trace()
# print(data.grad)
# print(offset.grad)
# print(weight.grad)
# print(bias.grad)
loss.backward()
# print(data.grad)
# print(cpudata.grad)
# print(bias.grad)