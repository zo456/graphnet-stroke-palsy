import torch
import torch.nn as nn
#import torch_geometric.nn as gnn
import torch.nn.functional as F
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool

class GVGG(nn.Module):
    def __init__(self, num_classes=2):
        super(GVGG, self).__init__()
        self.layer1 = nn.Sequential(
            GCNConv(2, 64),
            BatchNorm(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            GCNConv(64, 64),
            BatchNorm(64),
            nn.ReLU(),)
        self.layer3 = nn.Sequential(
            GCNConv(64, 128),
            BatchNorm(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            GCNConv(128, 128),
            BatchNorm(128),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            GCNConv(128, 256),
            BatchNorm(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            GCNConv(256, 256),
            BatchNorm(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            GCNConv(256, 256),
            BatchNorm(256),
            nn.ReLU())
        self.layer8 = nn.Sequential(
            GCNConv(256, 512),
            BatchNorm(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            GCNConv(512, 512),
            BatchNorm(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            GCNConv(512, 512),
            BatchNorm(512),
            nn.ReLU())
        self.layer11 = nn.Sequential(
            GCNConv(512, 512),
            BatchNorm(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            GCNConv(512, 512),
            BatchNorm(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            GCNConv(512, 512),
            BatchNorm(512),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x, ei, batch):
        out = self.layer1(x.float(), ei)
        out = self.layer2(out, ei)
        out = self.layer3(out, ei)
        out = self.layer4(out, ei)
        out = global_mean_pool(out, batch) 
        out = self.layer5(out, ei)
        out = self.layer6(out, ei)
        out = self.layer7(out, ei)
        out = global_mean_pool(out, batch) 
        out = self.layer8(out, ei)
        out = self.layer9(out, ei)
        out = self.layer10(out, ei)
        out = global_mean_pool(out, batch) 
        out = self.layer11(out, ei)
        out = self.layer12(out, ei)
        out = self.layer13(out, ei)
        #out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out