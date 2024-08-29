import torch
import torch.nn as  nn
import torch.nn.functional as F
import torch_geometric.nn.norm as gnnn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool, global_max_pool

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        
        self.conv1 = GCNConv(in_channels, out_channels)
        self.batch_norm1 = gnnn.BatchNorm(out_channels)
        
        self.conv2 = GCNConv(out_channels, out_channels)
        self.batch_norm2 = gnnn.BatchNorm(out_channels)
        
        self.conv3 = GCNConv(out_channels, out_channels*self.expansion)
        self.batch_norm3 = gnnn.BatchNorm(out_channels*self.expansion)
        self.relu = nn.ReLU()
        
    def forward(self, x, ei, batch):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x.float(), ei)))
        
        x = self.relu(self.batch_norm2(self.conv2(x, ei)))
        
        x = self.conv3(x, ei)
        x = self.batch_norm3(x)
        x+=identity
        x=self.relu(x)
        
        return x

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
       

        self.conv1 = GCNConv(in_channels, out_channels)
        self.batch_norm1 = gnnn.BatchNorm(out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)
        self.batch_norm2 = gnnn.BatchNorm(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, ei, batch):
      identity = x.clone()

      x = self.relu(self.batch_norm2(self.conv1(x.float(), ei)))
      x = self.batch_norm2(self.conv2(x, ei))

      print(x.shape)
      print(identity.shape)
      x += identity
      x = self.relu(x)
      return x

class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = GCNConv(2, 64)
        self.batch_norm1 = gnnn.BatchNorm(64)
        self.relu = nn.ReLU()
        self.max_pool = global_max_pool
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], 16)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], 16)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], 16)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], 16)
        
        self.avgpool = global_mean_pool
        self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
        
    def forward(self, x, ei, batch):
        x = self.relu(self.batch_norm1(self.conv1(x.float(), ei)))
        x = self.max_pool(x, batch)

        x = self.layer1(x, ei)
        x = self.layer2(x, ei)
        x = self.layer3(x, ei)
        x = self.layer4(x, ei)
        
        x = self.avgpool(x, batch)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, out_channls):
        layers = []
        layers.append(ResBlock(self.in_channels, out_channls))
        self.in_channels = out_channls*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, out_channls))
            
        return nn.Sequential(*layers)

        
        
def ResNet50(num_classes, channels=2):
    return ResNet(Bottleneck, [3,4,6,3], num_classes)