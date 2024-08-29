import sys
import argparse
import graph_load
import graph_model
from torch_geometric.nn.models import GCN
from torch.optim import Adam, RAdam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, fbeta_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt

device = 'cuda'
#data_dir = './tnf/'
data_dir = './yfp/'
#data_dir = '../celeba/'

parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--epoch', default=1000, type=int, help='number of epochs to train')
parser.add_argument('--lr', '--learning_rate', default=1e-2, type=float, help='initial learning rate')
parser.add_argument('--wd', '--weight_dacay', default=5e-5, type=float, help='weight decay')
parser.add_argument('--schedule', default=1000, type=int, help='Number of epochs to reduce LR')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--save_model', default='checkpoint-middy', type=str, help='Filename for saved model')
parser.add_argument('--tolerance', default = 6, type=int, help='Number of epochs before early stoppage')
parser.add_argument('--seed', default=3546, type=int, help='Random seed for all initialization')

args=parser.parse_args()

def main():
    for i in range(5):
        with LoggingPrinter(f'{args.save_model}_split_{i}.txt'):
            train_loader, val_loader, test_loader = graph_load.LoadGraphData(data_dir, args.batch_size, args.batch_size, i)
            torch.manual_seed(args.seed)
            #model= graph_model.GVGGDrop().to(device)
            #model = graph_model.GCNMid().to(device)
            model = graph_model.GCNMid().to(device)
            #model = graph_model.GCN2Layer().to(device)
            #model = graph_model.GVGG().to(device)
            #model = graph_model.GCN().to(device)
            #model = graph_model.GCNTiny().to(device)
            #model = graph_model.GAT().to(device)
            #model = GCN(in_channels=2, hidden_channels=64, num_layers=5, out_channels=64, dropout=0.3)
            print(f'Seed: {args.seed}, model: GCNMid')
            model.cuda()
            #model = torch.compile(model)
            optimizer = Adam(model.parameters(), lr = args.lr, weight_decay=args.wd)
            #optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=args.wd)
            #scheduler = StepLR(optimizer, step_size=args.schedule, gamma=0.7)
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, threshold=1e-2, patience=100, cooldown=50)
            check = 0
            track = 0
            old_lr = 0
            losses = []
            print("SPLIT:", i)
            for epoch in range(args.epoch):
                print("EPOCH:", epoch)
                if track < args.tolerance:
                    loss = train(train_loader, model, optimizer)
                    losses.append(loss.cpu().detach().numpy())
                    train_pred, train_target = validate(train_loader, model)
                    if epoch % 10000 == 0:
                        print(f"Train total: {train_target.cpu().shape}, Train positive: {train_target.cpu().sum()}")
                    train_acc = accuracy_score(train_target.cpu(), train_pred.cpu())
                    print(f"Training acc: {train_acc}")

                    pred_val, target_val = validate(val_loader, model)
                    print(f'Loss:{loss}')
                    print(f"Val total: {target_val.cpu().shape}, val positive: {target_val.cpu().sum()}")
                    print(f"Val acc: {accuracy_score(target_val.cpu(), pred_val.cpu())}")
                    print(f"Val F2: {fbeta_score(target_val.cpu(), pred_val.cpu(), beta=2.0)}")
                
                    scheduler.step(train_acc)
                    lr_now = optimizer.param_groups[0]['lr']
                    if lr_now != old_lr:
                        print(f'New LR: {lr_now}')
                        old_lr = lr_now

                    check = fbeta_score(target_val.cpu(), pred_val.cpu(), beta=2.0)
                    if check < 0.90:
                        track = 0
                    else:
                        track += 1
                elif accuracy_score(target_val.cpu(), pred_val.cpu()) < 0.90:
                    track = 0
                else:
                    print("Stop condition reached!")
                    break

            plt.plot(losses)
            plt.savefig(f'{args.save_model}_split_{i}.png')

            torch.save(model.state_dict(), f'{args.save_model}_split_{i}.pt')    

            pred_test, target_test = test(test_loader, model)
            print(f"Test total: {target_test.cpu().shape}, test positive: {target_test.cpu().sum()}")
            print(f"Test acc: {accuracy_score(target_test.cpu(), pred_test.cpu())}")
            print(f"Test F2: {fbeta_score(target_test.cpu(), pred_test.cpu(), beta=2.0)}")

def train(train_loader, model, optimizer):
    model.train()

    for data in train_loader:
        optimizer.zero_grad()
        data = data.cuda()
        out = model(data.x, data.edge_index, data.batch)
        loss = torch.nn.CrossEntropyLoss()(out, data.y)
        loss.backward()
        optimizer.step()
        return loss

def validate(val_loader, model):
    model.eval()
    pred_all = []
    target_all = []

    with torch.no_grad():
        for data in val_loader:
            data = data.cuda()
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            pred_all.append(pred)
            target_all.append(data.y)

        pred_all = torch.cat(pred_all, dim=0)
        target_all = torch.cat(target_all, dim=0)

        return pred_all, target_all
    
def test(test_loader, model):
    model.eval()
    pred_all = []
    target_all = []

    with torch.no_grad():
        for data in test_loader:
            data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            pred_all.append(pred)
            target_all.append(data.y)

        pred_all = torch.cat(pred_all, dim=0)
        target_all = torch.cat(target_all, dim=0)

        return pred_all, target_all

class LoggingPrinter:
    def __init__(self, filename):
        self.out_file = open(filename, "w")
        self.old_stdout = sys.stdout
        sys.stdout = self
    def write(self, text): 
        self.old_stdout.write(text)
        self.out_file.write(text)
    def __enter__(self): 
        return self
    def __exit__(self, type, value, traceback): 
        sys.stdout = self.old_stdout

if __name__ == '__main__':
    main()
