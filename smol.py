import argparse
import graph_load
import graph_model
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, fbeta_score, f1_score, precision_recall_curve

device = 'cuda'
data_dir = '../guoetal-arraydata-tnf/Data/'
#data_dir = '../celeba/'

parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--epoch', default=10, type=int, help='number of epochs to train')
parser.add_argument('--lr', '--learning_rate', default=1e-2, type=float, help='initial learning rate')
parser.add_argument('--wd', '--weight_dacay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--schedule', default=1000, type=int, help='Number of epochs to reduce LR')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--save_model', default='checkpoint.pt', type=str, help='Filename for saved model')

args=parser.parse_args()

def main():
    model = graph_model.GCN().to(device)
    train_loader, val_loader, test_loader = graph_load.LoadGraphData(data_dir, args.batch_size, args.batch_size)
    model.cuda()
    optimizer = Adam(model.parameters(), lr = args.lr, weight_decay=args.wd)
    scheduler = StepLR(optimizer, step_size=args.schedule, gamma=0.5)
    check = 0
    track = 0
    for epoch in range(args.epoch):
        print("EPOCH:", epoch)
        if track < 6:
            loss = train(train_loader, model, optimizer)
            train_pred, train_target = validate(train_loader, model)
            print(f"Training acc: {accuracy_score(train_target.cpu(), train_pred.cpu())}")

            pred_val, target_val = validate(val_loader, model)
            print(f'Loss:{loss}')
            #print(target_val.shape[0])
            #print(sum(target_val))
            #print(sum(pred_val))
            print(f"Val acc: {accuracy_score(target_val.cpu(), pred_val.cpu())}")
            check = accuracy_score(target_val.cpu(), pred_val.cpu())
            scheduler.step()
            if check < 0.9:
                track = 0
            else:
                track += 1
        else:
            break
    torch.save(model.state_dict(), f'./{args.save_model}')    
    pred_test, target_test = test(test_loader, model)
    print(f"Test acc: {accuracy_score(target_test.cpu(), pred_test.cpu())}")
   

def train(train_loader, model, optimizer):
    model.train()

    for data in train_loader:
        optimizer.zero_grad()
        data = data.cuda()
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
    
if __name__ == '__main__':
    main()
