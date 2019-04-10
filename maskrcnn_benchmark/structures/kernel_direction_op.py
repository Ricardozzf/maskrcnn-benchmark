import torch

xPos_matrix = torch.Tensor([[0,0,0.5],[0,0,1],[0,0,0.5]]).view(1,1,3,3)
xNeg_matrix = torch.Tensor([[0.5,0,0],[1,0,0],[0.5,0,0]]).view(1,1,3,3)
yPos_matrix = torch.Tensor([[0.5,1,0.5],[0,0,0],[0,0,0]]).view(1,1,3,3)
yNeg_matrix = torch.Tensor([[0,0,0],[0,0,0],[0.5,1,0.5]]).view(1,1,3,3)

Center_matrix = torch.Tensor([[-1,-1,-1],[-1,1,-1],[-1,-1,-1]]).view(1,1,3,3)

def kernel_direction(conv_kernel, conv_list,group=4):
    assert conv_kernel.shape[2:4] == torch.Size([3,3])
    kernel_num = conv_kernel.shape[0]
    kernel_channel = conv_kernel.shape[1]
    channel_stride = kernel_channel // group

    global xPos_matrix, xNeg_matrix, yPos_matrix, yNeg_matrix, Center_matrix

    xPos_matrix = xPos_matrix.to(conv_kernel.device)
    xNeg_matrix = xNeg_matrix.to(conv_kernel.device)
    yPos_matrix = yPos_matrix.to(conv_kernel.device)
    yNeg_matrix = yNeg_matrix.to(conv_kernel.device)
    Center_matrix = Center_matrix.to(conv_kernel.device)

    #import pdb; pdb.set_trace()
    conv_list[0].weight.data = (conv_kernel * xPos_matrix).sum((2,3)).view(kernel_num, kernel_channel, 1, 1) * conv_kernel
    conv_list[1].weight.data = (conv_kernel * xNeg_matrix).sum((2,3)).view(kernel_num, kernel_channel, 1, 1) * conv_kernel
    conv_list[2].weight.data = (conv_kernel * yPos_matrix).sum((2,3)).view(kernel_num, kernel_channel, 1, 1) * conv_kernel
    conv_list[3].weight.data = (conv_kernel * yNeg_matrix).sum((2,3)).view(kernel_num, kernel_channel, 1, 1) * conv_kernel

    '''
    kernel_center_xPos = (conv_kernel[:, :channel_stride, :, :] * Center_matrix).sum((2,3)).view(kernel_num, channel_stride, 1, 1) / 4
    kernel_center_xNeg = (conv_kernel[:, channel_stride:2*channel_stride, :, :] * Center_matrix).sum((2,3)).view(kernel_num, channel_stride, 1, 1) / 4
    kernel_center_yPos = (conv_kernel[:, 2*channel_stride:3*channel_stride, :, :] * Center_matrix).sum((2,3)).view(kernel_num, channel_stride, 1, 1) / 4
    kernel_center_yPos = (conv_kernel[:, 3*channel_stride:4*channel_stride, :, :] * Center_matrix).sum((2,3)).view(kernel_num, channel_stride, 1, 1) / 4
    '''
    

    #return torch.cat((xPos_tensor, xNeg_tensor, yPos_tensor, yNeg_tensor),dim=1)

