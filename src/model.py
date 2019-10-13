"""
Inspiration for the network was taken from the paper on Multi-View CNNS
Paper : http://vis-www.cs.umass.edu/mvcnn/docs/su15mvcnn.pdf
Github : https://github.com/RBirkeland/MVCNN-PyTorch
"""
## Dependencies

import torch
import torch.nn as nn
import torch.nn.functional as F


class MVCNN(nn.Module):
    
    def __init__(self, n_classes):
        
        super(MVCNN,self).__init__()
        pad = 1
        
        self.cnn = nn.Sequential(nn.BatchNorm2d(1),
                                     nn.Conv2d(1,32,3,padding=pad),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(32),
                                     nn.Conv2d(32,32,3,padding=pad),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2,2), 
        
                                     nn.BatchNorm2d(32),
                                     nn.Conv2d(32,64,3,padding=pad),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(64),
                                     nn.Conv2d(64,64,3,padding=pad),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2,2),
                                     
                                     nn.BatchNorm2d(64),
                                     nn.Conv2d(64,128,3,padding=pad),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(128),
                                     nn.Conv2d(128,128,3,padding=pad),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2,2),
        
                                     nn.BatchNorm2d(128),
                                     nn.Conv2d(128,256,3,padding=pad),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(256),
                                     nn.Conv2d(256,256,3,padding=pad),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2,2), 
        
                                     nn.BatchNorm2d(256),
                                     nn.Conv2d(256,256,3,padding=pad),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(256),
                                     nn.Conv2d(256,256,3,padding=pad),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2,2),
                                     
                                     nn.BatchNorm2d(256),
                                     nn.Conv2d(256,512,3,padding=pad),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(512),
                                     nn.Conv2d(512,512,3,padding=pad),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2,2))
  
        self.fc1 = nn.Sequential(nn.Linear(8192, 1096), 
                                     nn.ReLU(),
                                     nn.Dropout(0.8),
                                     nn.Linear(1096, 96),
                                     nn.ReLU(),
                                     nn.Dropout(0.9),
                                     nn.Linear(96, n_classes))


        
    def forward(self, x, batch_size, mvcnn=True):
        
        if mvcnn:
            view_pool = []
            # Assuming x has shape (x, 1, 299, 299)
            for n, v in enumerate(x):
                v = v.unsqueeze(0)
                v = self.cnn(v)
                v = v.view(v.size(0), 512 * 4 * 4)
                view_pool.append(v)

            pooled_view = view_pool[0]
            for i in range(1, len(view_pool)):
                pooled_view = torch.max(pooled_view, view_pool[i])

            output = self.fc1(pooled_view)
        
        else:
            x = self.cnn(x)
            x = x.view(-1, 512 * 4* 4)
            x = self.fc1(x)
            output = F.sigmoid(x)
    
        return output


def attention_block():
    
    return nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(1, 1, 1, padding=0),
        nn.BatchNorm2d(1),
        nn.Sigmoid()
    )


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True))


def one_conv(in_channels, padding=0):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.Conv2d(in_channels, 1, 1, padding=padding))


class UNetA(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
                
        self.dconv_down1 = double_conv(1, 32)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 128)
        self.dconv_down4 = double_conv(128, 256)        

        self.maxpool     = nn.MaxPool2d(2)
        self.upsample    = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.oneconv     = one_conv
        self.attention   = attention_block()
        
        self.oneconvx3 = one_conv(128)
        self.oneconvg3 = one_conv(256)
        self.dconv_up3 = double_conv(128 + 256, 128)
        
        self.oneconvx2 = one_conv(64)
        self.oneconvg2 = one_conv(128)
        self.dconv_up2 = double_conv(64 + 128, 64)
        
        
        self.conv_last = nn.Sequential(nn.BatchNorm2d(64),
                                     nn.Conv2d(64,32,3,padding=0),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2,2),
                                     nn.BatchNorm2d(32),
                                     nn.Conv2d(32,8,3,padding=0),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2,2))
        
        self.fc1 = nn.Sequential(nn.Linear(9800, 1096), 
                                     nn.ReLU(),
                                     nn.Dropout(0.8),
                                     nn.Linear(1096, 96),
                                     nn.ReLU(),
                                     nn.Dropout(0.9),
                                     nn.Linear(96, n_classes),
                                     nn.Softmax())
        
        
    def forward(self, array, batch_size, mvcnn=True):

        if mvcnn:
            view_pool = []
            for n, x in enumerate(array):
                #print(array.shape)
                x = x.unsqueeze(0)
                conv1 = self.dconv_down1(x) # 1 -> 32 filters
                x = self.maxpool(conv1)

                conv2 = self.dconv_down2(x) # 32 -> 64 filters
                x = self.maxpool(conv2)
                
                conv3 = self.dconv_down3(x) # 64 -> 128 filters
                x = self.maxpool(conv3)   
                
                x = self.dconv_down4(x)     # 128 -> 256 filters
                
                x = self.upsample(x)        
                _g = self.oneconvg3(x)
                _x = self.oneconvx3(conv3)
                _xg = _g + _x
                psi = self.attention(_xg)
                conv3 = conv3*psi
                x = torch.cat([x, conv3], dim=1) 
                
                x = self.dconv_up3(x)      # 128 + 256 -> 128 filters
                
                x = self.upsample(x)
                _g = self.oneconvg2(x)
                _x = self.oneconvx2(conv2)
                _xg = _g + _x
                psi = self.attention(_xg) 
                conv2 = conv2*psi
                x = torch.cat([x, conv2], dim=1)       

                x = self.dconv_up2(x)
                x = self.conv_last(x)

                x = x.view(-1, 35*35*8)
                view_pool.append(x)

            pooled_view = view_pool[0]
            for i in range(1, len(view_pool)):
                pooled_view = torch.max(pooled_view, view_pool[i])

            output = self.fc1(pooled_view)

        else:

            conv1 = self.dconv_down1(array) # 1 -> 32 filters
            x = self.maxpool(conv1)

            conv2 = self.dconv_down2(x)     # 32 -> 64 filters
            x = self.maxpool(conv2)
            
            conv3 = self.dconv_down3(x)     # 64 -> 128 filters
            x = self.maxpool(conv3)   
            
            x = self.dconv_down4(x)         # 128 -> 256 filters
            
            x = self.upsample(x)        
            _g = self.oneconvg3(x)
            _x = self.oneconvx3(conv3)
            _xg = _g + _x
            psi = self.attention(_xg)
            conv3 = conv3*psi
            x = torch.cat([x, conv3], dim=1) 
            
            x = self.dconv_up3(x)           # 128 + 256 -> 128 filters
            
            x = self.upsample(x)
            _g = self.oneconvg2(x)
            _x = self.oneconvx2(conv2)
            _xg = _g + _x
            psi = self.attention(_xg) 
            conv2 = conv2*psi
            x = torch.cat([x, conv2], dim=1)       

            x = self.dconv_up2(x)
            x = self.conv_last(x)

            x = x.view(-1, 35*35*8)
            output = self.fc1(x)

        return output

class UNet(nn.Module):

    def __init__(self):
        super().__init__()
                
        self.dconv_down1 = double_conv(1, 32)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 128)
        self.dconv_down4 = double_conv(128, 256)       

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(128 + 256, 128)
        self.dconv_up2 = double_conv(64 + 128, 64)
        self.dconv_up1 = double_conv(32 + 64, 32)
        
        self.conv_last = nn.Sequential(nn.BatchNorm2d(64),
                                     nn.Conv2d(64,32,3,padding=0),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2,2),
                                     nn.BatchNorm2d(32),
                                     nn.Conv2d(32,8,3,padding=0),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2,2))

        self.fc1 = nn.Sequential(nn.Linear(9800, 1096), 
                                     nn.ReLU(),
                                     nn.Dropout(0.8),
                                     nn.Linear(1096, 96),
                                     nn.ReLU(),
                                     nn.Dropout(0.9),
                                     nn.Linear(96, 20))
        
        
    def forward(self, array, batch_size, mvcnn=True):

        if mvcnn:

            view_pool = []
            for n, x in enumerate(array):

                x = x.unsqueeze(0)
                conv1 = self.dconv_down1(x)
                x = self.maxpool(conv1)

                conv2 = self.dconv_down2(x)
                x = self.maxpool(conv2)
                
                conv3 = self.dconv_down3(x)
                x = self.maxpool(conv3)   
                
                x = self.dconv_down4(x)
                
                x = self.upsample(x)        
                x = torch.cat([x, conv3], dim=1)
                
                x = self.dconv_up3(x)
                x = self.upsample(x)        
                x = torch.cat([x, conv2], dim=1)       

                x = self.dconv_up2(x)
                
                x = self.conv_last(x)

                x = x.view(-1, 35*35*8)
                view_pool.append(x)

            pooled_view = view_pool[0]
            for i in range(1, len(view_pool)):
                pooled_view = torch.max(pooled_view, view_pool[i])

            output = self.fc1(pooled_view)

        else:

            conv1 = self.dconv_down1(array)
            x = self.maxpool(conv1)

            conv2 = self.dconv_down2(x)
            x = self.maxpool(conv2)
            
            conv3 = self.dconv_down3(x)
            x = self.maxpool(conv3)   
            
            x = self.dconv_down4(x)
            
            x = self.upsample(x)        
            x = torch.cat([x, conv3], dim=1)
            
            x = self.dconv_up3(x)
            x = self.upsample(x)        
            x = torch.cat([x, conv2], dim=1)       

            x = self.dconv_up2(x)
            x = self.conv_last(x)
            x = x.view(-1, 35*35*8)

            output = self.fc1(x)
        
        return output