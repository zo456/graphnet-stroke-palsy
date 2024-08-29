import torch
#import torch_geometric.nn as gnn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.nn.norm import BatchNorm

class GCN2Layer(torch.nn.Module):
    def __init__(self):
        super(GCN2Layer, self).__init__()
        self.conv1 = GCNConv(2, 64)
        self.conv2 = GCNConv(64, 2)

    def forward(self, x, ei, batch):
        x = F.relu(self.conv1(x, ei))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, ei)
        x = global_max_pool(x, batch) 
        return F.log_softmax(x, dim=1)


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(2, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 128)
        self.conv4 = GCNConv(128, 128)
        self.conv5 = GCNConv(128, 64)
        self.lin1 = torch.nn.Linear(64, 10)
        self.lin = torch.nn.Linear(10, 2)
        self.nonlin = torch.nn.GELU()
        self.bn1 = BatchNorm(64)
        self.bn2 = BatchNorm(128)
        self.bn3 = BatchNorm(64)

    def forward(self, x, ei, batch):

        x = self.conv1(x.float(), ei)
        x = self.bn1(x)
        x = self.nonlin(x)
        #x = F.leaky_relu(x)
        x = self.conv2(x, ei)
        x = self.nonlin(x)
        #x = F.leaky_relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, ei)
        x = self.bn2(x)
        x = self.nonlin(x)
        x = self.conv4(x, ei)
        x = self.nonlin(x)
        x = self.conv5(x, ei)
        x = self.bn3(x)
        x = self.nonlin(x)
        x = global_max_pool(x, batch) 
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = self.lin(x)

        return x
    
class GCNTiny(torch.nn.Module):
    def __init__(self):
        super(GCNTiny, self).__init__()
        self.conv1 = GCNConv(2, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)
        self.conv4 = GCNConv(32, 32)
        self.conv5 = GCNConv(32, 32)
        self.lin1 = torch.nn.Linear(32, 10)
        self.lin = torch.nn.Linear(10, 2)
        self.nonlin = torch.nn.GELU()
        self.bn1 = BatchNorm(32)
        self.bn2 = BatchNorm(32)
        self.bn3 = BatchNorm(32)

    def forward(self, x, ei, batch):

        x = self.conv1(x.float(), ei)
        x = self.bn1(x)
        x = self.nonlin(x)
        #x = F.leaky_relu(x)
        x = self.conv2(x, ei)
        x = self.nonlin(x)
        #x = F.leaky_relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, ei)
        x = self.bn2(x)
        x = self.nonlin(x)
        x = self.conv4(x, ei)
        x = self.nonlin(x)
        x = self.conv5(x, ei)
        x = self.bn3(x)
        x = self.nonlin(x)
        x = global_max_pool(x, batch) 
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = self.lin(x)

        return x
    
class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.conv1 = GATConv(2, 64)
        self.conv2 = GATConv(64, 64)
        self.conv3 = GATConv(64, 128)
        self.conv4 = GATConv(128, 256)
        self.conv5 = GATConv(256, 256)
        self.lin1 = torch.nn.Linear(256, 10)
        self.lin = torch.nn.Linear(10, 2)
        self.nonlin = torch.nn.GELU()
        self.bn1 = BatchNorm(64)
        self.bn2 = BatchNorm(128)
        self.bn3 = BatchNorm(256)

    def forward(self, x, ei, batch):

        x = self.conv1(x.float(), ei)
        x = self.bn1(x)
        x = self.nonlin(x)
        #x = F.leaky_relu(x)
        x = self.conv2(x, ei)
        x = self.nonlin(x)
        #x = F.leaky_relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, ei)
        x = self.bn2(x)
        x = self.nonlin(x)
        x = self.conv4(x, ei)
        x = self.nonlin(x)
        x = self.conv5(x, ei)
        x = self.bn3(x)
        x = self.nonlin(x)
        x = global_max_pool(x, batch) 
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = self.lin(x)

        return x
    
class GCNMid(torch.nn.Module):
    def __init__(self):
        super(GCNMid, self).__init__()
        self.conv1 = GCNConv(2, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 128)
        self.conv4 = GCNConv(128, 256)
        self.conv5 = GCNConv(256, 256)
        self.lin1 = torch.nn.Linear(256, 10)
        #self.lin2 = torch.nn.Linear(256, 10)
        self.lin = torch.nn.Linear(10, 2)
        self.nonlin = torch.nn.GELU()
        self.bn1 = BatchNorm(64)
        self.bn2 = BatchNorm(128)
        self.bn3 = BatchNorm(256)

    def forward(self, x, ei, batch):

        x = self.conv1(x.float(), ei)
        x = self.bn1(x)
        x = self.nonlin(x)
        #x = F.leaky_relu(x)
        x = self.conv2(x, ei)
        x = self.nonlin(x)
        #x = F.leaky_relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, ei)
        x = self.bn2(x)
        x = self.nonlin(x)
        x = self.conv4(x, ei)
        x = self.nonlin(x)
        x = self.conv5(x, ei)
        x = self.bn3(x)
        x = self.nonlin(x)
        x = global_max_pool(x, batch) 
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin1(x)
        #x = self.lin2(x)
        x = self.lin(x)

        return x
    
class GCNOth(torch.nn.Module):
    def __init__(self):
        super(GCNOth, self).__init__()
        self.conv1 = GCNConv(2, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 128)
        self.conv4 = GCNConv(128, 128)
        self.conv5 = GCNConv(128, 256)
        self.lin1 = torch.nn.Linear(256, 10)
        #self.lin2 = torch.nn.Linear(256, 10)
        self.lin = torch.nn.Linear(10, 2)
        self.nonlin = torch.nn.GELU()
        self.bn1 = BatchNorm(64)
        self.bn2 = BatchNorm(128)
        self.bn3 = BatchNorm(256)

    def forward(self, x, ei, batch):

        x = self.conv1(x.float(), ei)
        x = self.bn1(x)
        x = self.nonlin(x)
        #x = F.leaky_relu(x)
        x = self.conv2(x, ei)
        x = self.nonlin(x)
        #x = F.leaky_relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, ei)
        x = self.bn2(x)
        x = self.nonlin(x)
        x = self.conv4(x, ei)
        x = self.nonlin(x)
        x = self.conv5(x, ei)
        x = self.bn3(x)
        x = self.nonlin(x)
        x = global_mean_pool(x, batch) 
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin1(x)
        #x = self.lin2(x)
        x = self.lin(x)

        return x
    
class GVGG(torch.nn.Module):
    def __init__(self):
        super(GVGG, self).__init__()
        self.conv1 = GCNConv(2, 64)
        self.bn1 = BatchNorm(64)
        self.conv2 = GCNConv(64, 64)
        self.bn2 = BatchNorm(64)
        self.conv3 = GATConv(64, 128)
        self.bn3 = BatchNorm(128)
        self.conv4 = GATConv(128, 128)
        self.bn4 = BatchNorm(128)
        self.conv5 = GATConv(128, 256)
        self.bn5 = BatchNorm(256)
        self.conv6 = GATConv(256, 256)
        self.bn6 = BatchNorm(256)
        self.conv7 = GATConv(256, 256)
        self.bn7 = BatchNorm(256)
        self.conv8 = GATConv(256, 512)
        self.bn8 = BatchNorm(512)
        self.conv9 = GATConv(512, 512)
        self.bn9 = BatchNorm(512)
        self.conv10 = GATConv(512, 512)
        self.bn10 = BatchNorm(512)
        self.conv11 = GATConv(512, 512)
        self.bn11 = BatchNorm(512)
        self.conv12 = GATConv(512, 512)
        self.bn12 = BatchNorm(512)
        self.conv13 = GATConv(512, 512)
        self.bn13 = BatchNorm(512)
        self.fc = torch.nn.Linear(512, 4096)
        self.fc1 = torch.nn.Linear(4096, 4096)
        self.fc2 = torch.nn.Linear(4096, 2)

    def forward(self, x, ei, batch):
        x = self.conv1(x.float(), ei)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x, ei)
        x = self.bn2(x)
        x = F.relu(x)
        #x = global_max_pool(x, batch) 
        
        x = self.conv3(x, ei)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.conv4(x, ei)
        x = self.bn4(x)
        x = F.relu(x)
        #x = global_max_pool(x, batch) 
        
        x = self.conv5(x, ei)
        x = self.bn5(x)
        x = F.relu(x)
        
        x = self.conv6(x, ei)
        x = self.bn6(x)
        x = F.relu(x)
        
        x = self.conv7(x, ei)
        x = self.bn7(x)
        x = F.relu(x)
        #x = global_max_pool(x, batch) 
        
        x = self.conv8(x, ei)
        x = self.bn8(x)
        x = F.relu(x)
        
        x = self.conv9(x, ei)
        x = self.bn9(x)
        x = F.relu(x)
        
        x = self.conv10(x, ei)
        x = self.bn10(x)
        x = F.relu(x)
        #x = global_max_pool(x, batch) 
        
        x = self.conv11(x, ei)
        x = self.bn11(x)
        x = F.relu(x)
        
        x = self.conv12(x, ei)
        x = self.bn12(x)
        x = F.relu(x)
        
        x = self.conv13(x, ei)
        x = self.bn13(x)
        x = F.relu(x)
        x = global_max_pool(x, batch) 

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        x = F.relu(x)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x
    

class GVGGDrop(torch.nn.Module):
    def __init__(self):
        super(GVGGDrop, self).__init__()
        self.conv1 = GCNConv(2, 64)
        self.bn1 = BatchNorm(64)
        self.conv2 = GCNConv(64, 64)
        self.bn2 = BatchNorm(64)
        self.conv3 = GATConv(64, 128)
        self.bn3 = BatchNorm(128)
        self.conv4 = GATConv(128, 128)
        self.bn4 = BatchNorm(128)
        self.conv5 = GATConv(128, 256)
        self.bn5 = BatchNorm(256)
        self.conv6 = GATConv(256, 256)
        self.bn6 = BatchNorm(256)
        self.conv7 = GATConv(256, 256)
        self.bn7 = BatchNorm(256)
        self.conv8 = GATConv(256, 512)
        self.bn8 = BatchNorm(512)
        self.conv9 = GATConv(512, 512)
        self.bn9 = BatchNorm(512)
        self.conv10 = GATConv(512, 512)
        self.bn10 = BatchNorm(512)
        self.conv11 = GATConv(512, 512)
        self.bn11 = BatchNorm(512)
        self.conv12 = GATConv(512, 512)
        self.bn12 = BatchNorm(512)
        self.conv13 = GATConv(512, 512)
        self.bn13 = BatchNorm(512)
        self.fc = torch.nn.Linear(512, 4096)
        self.fc1 = torch.nn.Linear(4096, 4096)
        self.fc2 = torch.nn.Linear(4096, 2)

    def forward(self, x, ei, batch):
        x = self.conv1(x.float(), ei)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x, ei)
        x = self.bn2(x)
        x = F.relu(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        #x = global_max_pool(x, batch) 
        
        x = self.conv3(x, ei)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.conv4(x, ei)
        x = self.bn4(x)
        x = F.relu(x)
        #x = global_max_pool(x, batch) 
        
        x = self.conv5(x, ei)
        x = self.bn5(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.15, training=self.training)
        
        x = self.conv6(x, ei)
        x = self.bn6(x)
        x = F.relu(x)
        
        x = self.conv7(x, ei)
        x = self.bn7(x)
        x = F.relu(x)
        #x = global_max_pool(x, batch) 
        
        x = self.conv8(x, ei)
        x = self.bn8(x)
        x = F.relu(x)
        
        x = self.conv9(x, ei)
        x = self.bn9(x)
        x = F.relu(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv10(x, ei)
        x = self.bn10(x)
        x = F.relu(x)
        #x = global_max_pool(x, batch) 
        
        x = self.conv11(x, ei)
        x = self.bn11(x)
        x = F.relu(x)
        
        x = self.conv12(x, ei)
        x = self.bn12(x)
        x = F.relu(x)
        
        x = self.conv13(x, ei)
        x = self.bn13(x)
        x = F.relu(x)
        x = global_max_pool(x, batch) 

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        x = F.relu(x)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x