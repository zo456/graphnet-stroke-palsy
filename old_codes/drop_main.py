import argparse
import graph_load
import graph_model
from torch.optim import Adam
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, fbeta_score, f1_score, precision_recall_curve

device = 'cuda'
data_dir = '../guoetal-arraydata-tnf/Data/'

parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--epoch', default=10, type=int, help='number of epochs to train')
parser.add_argument('--lr', '--learning_rate', default=1e-2, type=float, help='initial learning rate')
parser.add_argument('--wd', '--weight_dacay', default=5e-4, type=float, help='weight decay')

args=parser.parse_args()

def main():
    model = graph_model.GVGGDrop().to(device)
    train_loader, val_loader, test_loader = graph_load.LoadGraphData(data_dir, 128, 128)
    model.cuda()
    optimizer = Adam(model.parameters(), lr = args.lr, weight_decay=args.wd)
    for epoch in range(args.epoch):
        print("EPOCH:", epoch)
        loss = train(train_loader, model, optimizer)
        train_pred, train_target = validate(train_loader, model)
        print(f"Training acc: {accuracy_score(train_target.cpu(), train_pred.cpu())}")

        pred_val, target_val = validate(val_loader, model)
        print(f'Loss:{loss}')
        #print(target_val.shape[0])
        #print(sum(target_val))
        #print(sum(pred_val))
        print(f"Val acc: {accuracy_score(target_val.cpu(), pred_val.cpu())}")

    pred_test, target_test = test(test_loader, model)
    torch.save(model.state_dict(), './checkpoint-dropout0.pt')    
    print(f"Test acc: {accuracy_score(target_test.cpu(), pred_test.cpu())}")
    print(f"Test prec: {precision_score(target_test.cpu(), pred_test.cpu())}")
    print(f"Test rec: {recall_score(target_test.cpu(), pred_test.cpu())}")
    print(f"Test F1: {f1_score(target_test.cpu(), pred_test.cpu())}")  
    print(f"Test F2: {fbeta_score(target_test.cpu(), pred_test.cpu(), beta=2.0)}") 
   

def train(train_loader, model, optimizer):
    model.train()

    for data in train_loader:
        optimizer.zero_grad()
        data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        #pred = out.argmax(dim=1)
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
            data.to(device)
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
    
if __name__ == '__main__':
    main()
