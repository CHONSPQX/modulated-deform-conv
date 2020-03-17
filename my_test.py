from modulated_deform_conv import *
batch=1
cpudata=torch.ones(batch,1,5,5,requires_grad=True)
# data=torch.ones(batch,1,5,5,device='cuda',requires_grad=True)
data=cpudata.cuda()
offset=torch.zeros(batch,18,5,5,device='cuda',requires_grad=True)
mask=torch.ones(batch,9,5,5,device='cuda',requires_grad=True)
weight=torch.ones(1,1,3,3,device='cuda',requires_grad=True)
bias=torch.zeros(1,device='cuda',requires_grad=True)
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
print(data)
out=deform_conv2d(data,offset,weight,bias,stride,padding,dilation,groups,deformable_groups,in_step)
out=modulated_deform_conv2d(data,offset,mask,weight,bias,stride,padding,dilation,groups,deformable_groups,in_step)
print(out)

loss=out.sum()
print(loss)
print(data.grad)
print(offset.grad)
print(weight.grad)
print(bias.grad)
loss.backward()
print(data.grad)
print(cpudata.grad)
print(bias.grad)