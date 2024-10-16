import torch.nn as nn
import torch.nn.functional as F


class CNN_digitrec(nn.Module): #batch size does not matter for size, due to tensors. 
    def __init__(self):
        super().__init__()
        
        self.conv = nn.Conv2d(1, 32, kernel_size=3)
        self.lin = nn.Linear(26*26*32, 10)

    def forward(self, x):
        # conv layer 
        x = self.conv(x)
        x = F.relu(x)

        # lin layer 
        x = x.view(-1, 26*26*32) 
        x = self.lin(x)
        return x   

