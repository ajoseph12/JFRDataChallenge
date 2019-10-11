"""
Inspiration for the network was taken from the paper on Multi-View CNNS
Paper : http://vis-www.cs.umass.edu/mvcnn/docs/su15mvcnn.pdf
Github : https://github.com/RBirkeland/MVCNN-PyTorch
"""
## Dependencies

import torch
import torch.nn as nn

class MVCNN(nn.Module):
    
    def __init__(self):
        
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
  
        self.fc1 = nn.Sequential(nn.Linear(8192, 1024), 
                                     nn.ReLU(),
                                     nn.Dropout(0.8),
                                     nn.Linear(1024, 96),
                                     nn.ReLU(),
                                     nn.Dropout(0.9),
                                     nn.Linear(96, 1))

        self.fc2 = nn.Sequential(nn.Linear(8192, 4096), 
                                     nn.ReLU(),
                                     nn.Dropout(0.8),
                                     nn.Linear(4096, 4096),
                                     nn.ReLU(),
                                     nn.Dropout(0.9),
                                     nn.Linear(4096, 12),
                                     nn.Sigmoid())
        
    def forward(self, x, batch_size, mvcnn=True):
        
        if mvcnn:
            view_pool = []
            # Assuming x has shape (x, 1, 299, 299)
            for n, v in enumerate(x):
                v = v.unsqueeze(0)
                v = self.cnn(v)
                v = v.view(v.size(0), 512 * 4* 4)
                view_pool.append(v)

            pooled_view = view_pool[0]
            for i in range(1, len(view_pool)):
                pooled_view = torch.max(pooled_view, view_pool[i])

            output = self.fc1(pooled_view)
        
        else:
            x = self.cnn(x)
            x = x.view(-1, 512 * 4* 4)
            output = self.fc2(x)
    
        return output

