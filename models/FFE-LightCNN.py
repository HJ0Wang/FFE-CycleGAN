'''
    implement Light CNN
    @author: Alfred Xiang Wu
    @date: 2017.07.04

'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(
                in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)

        return torch.max(out[0], out[1])


class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv = mfm(in_channels, out_channels,
                        kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)

        return x


class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.conv1 = mfm(in_channels, out_channels,
                         kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(in_channels, out_channels,
                         kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res

        return out


class network_9layers(nn.Module):
    def __init__(self, num_classes=79077):
        super(network_9layers, self).__init__()
        self.features = nn.Sequential(
            mfm(1, 48, 5, 1, 2),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(48, 96, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(96, 192, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(192, 128, 3, 1, 1),
            group(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )
        self.fc1 = mfm(8*8*128, 256, type=0)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        out = self.fc2(x)
        return out, x


class network_29layers(nn.Module):
    def __init__(self, block, layers, num_classes=79077):
        super(network_29layers, self).__init__()
        self.conv1 = mfm(1, 48, 5, 1, 2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block1 = self._make_layer(block, layers[0], 48, 48)
        self.group1 = group(48, 96, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block2 = self._make_layer(block, layers[1], 96, 96)
        self.group2 = group(96, 192, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block3 = self._make_layer(block, layers[2], 192, 192)
        self.group3 = group(192, 128, 3, 1, 1)
        self.block4 = self._make_layer(block, layers[3], 128, 128)
        self.group4 = group(128, 128, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.fc = mfm(8*8*128, 256, type=0)
        self.fc2 = nn.Linear(256, num_classes)

    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.block1(x)
        x = self.group1(x)
        x = self.pool2(x)

        x = self.block2(x)
        x = self.group2(x)
        x = self.pool3(x)

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = self.pool4(x)

        x = x.view(x.size(0), -1)
        fc = self.fc(x)
        fc = F.dropout(fc, training=self.training)
        out = self.fc2(fc)
        return out, fc


class network_29layers_v2(nn.Module):
    def __init__(self, block, layers, num_classes=80013, feature=True):
        super(network_29layers_v2, self).__init__()
        self.conv1 = mfm(1, 48, 5, 1, 2)
        self.block1 = self._make_layer(block, layers[0], 48, 48)
        self.group1 = group(48, 96, 3, 1, 1)
        self.block2 = self._make_layer(block, layers[1], 96, 96)
        self.group2 = group(96, 192, 3, 1, 1)
        self.block3 = self._make_layer(block, layers[2], 192, 192)
        self.group3 = group(192, 128, 3, 1, 1)
        self.block4 = self._make_layer(block, layers[3], 128, 128)
        self.group4 = group(128, 128, 3, 1, 1) # -> 8*8*128
        self.fc = nn.Linear(8 * 8 * 128, 256)
        self.fc2 = nn.Linear(256, num_classes, bias=False)
        self.feature = feature

    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def makegray(self, input):
        x = input[:, 0, :, :] * 0.299 + input[:, 1, :, :] * \
            0.587 + input[:, 2, :, :] * 0.114  # to grayscale
        x = x.unsqueeze(1)
        return x

    def forward(self, input):
        # input: [B, 3 (r,g,b), 128, 128]PIL
        x = self.makegray(input)

        # expected x: [4, 1, 128, 128]
        x = self.conv1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block1(x)
        x = self.group1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block2(x)
        x = self.group2(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = x.view(x.size(0), -1)
        fc = self.fc(x)
        if self.feature:
            return fc
        x = F.dropout(fc, training=self.training)
        out = self.fc2(x)
        return out




# self-defined
class network_29layers_v3(nn.Module):
    def __init__(self, block, layers, num_classes=80013, feature=True):
        super(network_29layers_v2, self).__init__()
        self.conv1 = mfm(1, 48, 5, 1, 2)
        self.block1 = self._make_layer(block, layers[0], 48, 48)
        self.group1 = group(48, 96, 3, 1, 1)
        self.block2 = self._make_layer(block, layers[1], 96, 96)
        self.group2 = group(96, 192, 3, 1, 1)
        self.block3 = self._make_layer(block, layers[2], 192, 192)
        self.group3 = group(192, 128, 3, 1, 1)
        self.block4 = self._make_layer(block, layers[3], 128, 128)
        self.group4 = group(128, 128, 3, 1, 1) 
        self.fc = nn.Linear(8 * 8 * 128, 256)
        self.fc2 = nn.Linear(256, num_classes, bias=False)
        self.feature = feature

    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def makegray(self, input):
        x = input[:, 0, :, :] * 0.299 + input[:, 1, :, :] * \
            0.587 + input[:, 2, :, :] * 0.114  # to grayscale
        x = x.unsqueeze(1)
        return x

    def forward(self, input):
        # input: [B, 3 (r,g,b), 128, 128]PIL
        # ffe input [B, 3 (r,g,b), 256, 256]PIL, [B,c,h,w]
        x = self.makegray(input)

        # expected x: [4, 1, 128, 128]
        # ffe expected x: [4, 1, 256, 256]
        x = self.conv1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block1(x)
        x = self.group1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block2(x)
        x = self.group2(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
        # -> [4, 128, 32, 32] 

        # x = x.view(x.size(0), -1)
        # fc = self.fc(x)
        # if self.feature:
        #     return fc
        # x = F.dropout(fc, training=self.training)
        # out = self.fc2(x)
        out = x
        return out


def LightCNN_29Layers_v2(**kwargs):
    model = network_29layers_v2(resblock, [1, 2, 3, 4], **kwargs)
    return model


def define_R(gpu_ids, lightcnn_path):
    # import models.modules.arch_lightcnn as arch
    # netR = arch.LightCNN_29Layers_v2()
    netR = LightCNN_29Layers_v2()
    netR.eval()
    if gpu_ids:
        netR = torch.nn.DataParallel(netR).cuda()
    checkpoint = torch.load(lightcnn_path)
    netR.load_state_dict(checkpoint['state_dict'])

    return netR

def LightCNN_29Layers_FFE(**kwargs):
    model = network_29layers_v3(resblock, [1, 2, 3, 4], **kwargs)
    return model



def define_FFE_LightCNN(gpu_ids, lightcnn_path):
    # import models.modules.arch_lightcnn as arch
    # netR = arch.LightCNN_29Layers_v2()
    netR = LightCNN_29Layers_FFE()
    netR.eval()
    if gpu_ids:
        netR = torch.nn.DataParallel(netR).cuda()
    checkpoint = torch.load(lightcnn_path)
    netR.load_state_dict(checkpoint['state_dict'])

    return netR

    # elif opt.ffe_model == 'lightcnn':
    #     # get pretrained weights from mobilefacenet
    #     G_A_net_state = model.get_state_dict(GorN_name='G_A',printornot=1)
    #     G_B_net_state = model.get_state_dict(GorN_name='G_B',printornot=0)

    #     LightCNN = LightCNN_29Layers_FFE()
    #     LightCNN.eval()
    #     pretrained_dict = torch.load(opt.ffe_pretrained_weights)
    #     for key in pretrained_dict['state_dict']:
    #         print(key)
    #         key =  key.replace('module.','')
    #     LightCNN.load_state_dict(pretrained_dict['state_dict'])

    #     # for param_tensor in LightCNN.state_dict():  # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
    #     #     print(param_tensor, '\t', LightCNN.state_dict()[param_tensor].size())

    #     # # lightcnn key: module.conv1.filter.weight -> cyclegan key: model.0.conv1.filter.weight
    #     # pre_cycmbf_key = 'model.0.' 
    #     # LightCNN_dict = {}
    #     # for param_tensor in LightCNN.state_dict():
    #     #     param_tensor = param_tensor.replace('module.', pre_cycmbf_key)
    #     #     if param_tensor in G_A_net_state:
    #     #         print('replaced lightcnn key: ', param_tensor)
    #     #         LightCNN_dict[param_tensor] = LightCNN.state_dict()[param_tensor]

    #     # for key in pretrained_dict['state_dict']:
    #     #     replaced = key.replace('module.', pre_cycmbf_key)
    #     #     print(f"key in lightcnn: {key}, replaced：{replaced}")
    #     # for k,v in pretrained_dict.items():
    #     #     if k.replace('module.', pre_cycmbf_key) in G_A_net_state:
    #     #         print('replaces key: ', k.replace('module.', pre_cycmbf_key))
    #     #     else:
    #     #         print(f'k not in G_A_net_state, k: {k}')
    #     # pretrained_dict = {k.replace('module.',pre_cycmbf_key):v for k,v in pretrained_dict.items() if k.replace('module.', pre_cycmbf_key) in G_A_net_state}

    #     # G_A_net_state.update(pretrained_dict)
    #     # G_B_net_state.update(pretrained_dict)
    #     # model.save_networks_singlely(epoch='init', GorN_name='G_A',state_dict=G_A_net_state)
    #     # model.load_state_singlely(epoch='init', GorN_name='G_A')
    #     # model.save_networks_singlely(epoch='init', GorN_name='G_B',state_dict=G_A_net_state)
    #     # model.load_state_singlely(epoch='init', GorN_name='G_B')


# model = define_R(gpu_ids=[0], 
#                 lightcnn_path='/ssd01/wanghuijiao/CG/LightCNN29/LightCNN_29Layers_V2_checkpoint.pth')
# for parameters in model.parameters(): #打印出参数矩阵及值
#     print(parameters.size())
# for param_tensor in model.state_dict():  # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
#     print(param_tensor, '\t', model.state_dict()[param_tensor].size())


# # # input = torch.rand((4,3,256,256))
# # model = define_FFE_LightCNN(gpu_ids=[0], 
# #                             lightcnn_path='/ssd01/wanghuijiao/CG/LightCNN29/LightCNN_29Layers_V2_checkpoint.pth')

# ffe_model = LightCNN_29Layers_FFE()
# net_state = ffe_model.get_state_dict()

# pretrained_model = torch.load('/ssd01/wanghuijiao/CG/LightCNN29/LightCNN_29Layers_V2_checkpoint.pth')
# for k,v in pretrained_model.items():
#     print('k: v', k)

# G_A_net_state = model.get_state_dict(GorN_name='G_A',printornot=0)
# G_B_net_state = model.get_state_dict(GorN_name='G_B',printornot=0)

# pretrained_dict = torch.load('/ssd01/wanghuijiao/CG/LightCNN29/LightCNN_29Layers_V2_checkpoint.pth')
# # cyclegan key: model.0.conv_4.model.1.conv.prelu.weight, and mbfnet key: conv_4.model.1.conv.prelu.weight
# pre_cycmbf_key = '' #'model.0.' 
# pretrained_dict = {(pre_cycmbf_key + k):v for k,v in pretrained_dict.items() if (pre_cycmbf_key + k) in G_A_net_state}

# G_A_net_state.update(pretrained_dict)
# G_B_net_state.update(pretrained_dict)
# model.save_networks_singlely(epoch='init', GorN_name='G_A',state_dict=G_A_net_state)
# model.load_state_singlely(epoch='init', GorN_name='G_A')
# model.save_networks_singlely(epoch='init', GorN_name='G_B',state_dict=G_A_net_state)
# model.load_state_singlely(epoch='init', GorN_name='G_B')
# for parameters in model.parameters(): #打印出参数矩阵及值
#     print(parameters)




# for name, parameters in model.named_parameters():#打印出每一层的参数的大小
#     print(name, ':', parameters.size())

# for param_tensor in model.state_dict():  # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
#     print(param_tensor, '\t', model.state_dict()[param_tensor].size())

# output = model(input)
# print('output.shape: ', output.shape)


# module.conv1.filter.weight       torch.Size([96, 1, 5, 5])
# module.conv1.filter.bias         torch.Size([96])
# module.block1.0.conv1.filter.weight      torch.Size([96, 48, 3, 3])
# module.block1.0.conv1.filter.bias        torch.Size([96])
# module.block1.0.conv2.filter.weight      torch.Size([96, 48, 3, 3])
# module.block1.0.conv2.filter.bias        torch.Size([96])
# module.group1.conv_a.filter.weight       torch.Size([96, 48, 1, 1])
# module.group1.conv_a.filter.bias         torch.Size([96])
# module.group1.conv.filter.weight         torch.Size([192, 48, 3, 3])
# module.group1.conv.filter.bias   torch.Size([192])
# module.block2.0.conv1.filter.weight      torch.Size([192, 96, 3, 3])
# module.block2.0.conv1.filter.bias        torch.Size([192])
# module.block2.0.conv2.filter.weight      torch.Size([192, 96, 3, 3])
# module.block2.0.conv2.filter.bias        torch.Size([192])
# module.block2.1.conv1.filter.weight      torch.Size([192, 96, 3, 3])
# module.block2.1.conv1.filter.bias        torch.Size([192])
# module.block2.1.conv2.filter.weight      torch.Size([192, 96, 3, 3])
# module.block2.1.conv2.filter.bias        torch.Size([192])
# module.group2.conv_a.filter.weight       torch.Size([192, 96, 1, 1])
# module.group2.conv_a.filter.bias         torch.Size([192])
# module.group2.conv.filter.weight         torch.Size([384, 96, 3, 3])
# module.group2.conv.filter.bias   torch.Size([384])
# module.block3.0.conv1.filter.weight      torch.Size([384, 192, 3, 3])
# module.block3.0.conv1.filter.bias        torch.Size([384])
# module.block3.0.conv2.filter.weight      torch.Size([384, 192, 3, 3])
# module.block3.0.conv2.filter.bias        torch.Size([384])
# module.block3.1.conv1.filter.weight      torch.Size([384, 192, 3, 3])
# module.block3.1.conv1.filter.bias        torch.Size([384])
# module.block3.1.conv2.filter.weight      torch.Size([384, 192, 3, 3])
# module.block3.1.conv2.filter.bias        torch.Size([384])
# module.block3.2.conv1.filter.weight      torch.Size([384, 192, 3, 3])
# module.block3.2.conv1.filter.bias        torch.Size([384])
# module.block3.2.conv2.filter.weight      torch.Size([384, 192, 3, 3])
# module.block3.2.conv2.filter.bias        torch.Size([384])
# module.group3.conv_a.filter.weight       torch.Size([384, 192, 1, 1])
# module.group3.conv_a.filter.bias         torch.Size([384])
# module.group3.conv.filter.weight         torch.Size([256, 192, 3, 3])
# module.group3.conv.filter.bias   torch.Size([256])
# module.block4.0.conv1.filter.weight      torch.Size([256, 128, 3, 3])
# module.block4.0.conv1.filter.bias        torch.Size([256])
# module.block4.0.conv2.filter.weight      torch.Size([256, 128, 3, 3])
# module.block4.0.conv2.filter.bias        torch.Size([256])
# module.block4.1.conv1.filter.weight      torch.Size([256, 128, 3, 3])
# module.block4.1.conv1.filter.bias        torch.Size([256])
# module.block4.1.conv2.filter.weight      torch.Size([256, 128, 3, 3])
# module.block4.1.conv2.filter.bias        torch.Size([256])
# module.block4.2.conv1.filter.weight      torch.Size([256, 128, 3, 3])
# module.block4.2.conv1.filter.bias        torch.Size([256])
# module.block4.2.conv2.filter.weight      torch.Size([256, 128, 3, 3])
# module.block4.2.conv2.filter.bias        torch.Size([256])
# module.block4.3.conv1.filter.weight      torch.Size([256, 128, 3, 3])
# module.block4.3.conv1.filter.bias        torch.Size([256])
# module.block4.3.conv2.filter.weight      torch.Size([256, 128, 3, 3])
# module.block4.3.conv2.filter.bias        torch.Size([256])
# module.group4.conv_a.filter.weight       torch.Size([256, 128, 1, 1])
# module.group4.conv_a.filter.bias         torch.Size([256])
# module.group4.conv.filter.weight         torch.Size([256, 128, 3, 3])
# module.group4.conv.filter.bias   torch.Size([256])
# module.fc.weight         torch.Size([256, 8192])
# module.fc.bias   torch.Size([256])
# module.fc2.weight        torch.Size([80013, 256])
