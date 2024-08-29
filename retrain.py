import argparse
import graph_load
import graph_model
from torch.optim import Adam, RAdam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, fbeta_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt

device = 'cuda'
data_dir = '../guoetal-arraydata-tnf/Data/'
#data_dir = '../celeba/'

parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--epoch', default=10, type=int, help='number of epochs to train')
parser.add_argument('--lr', '--learning_rate', default=1e-2, type=float, help='initial learning rate')
parser.add_argument('--wd', '--weight_dacay', default=5e-5, type=float, help='weight decay')
parser.add_argument('--schedule', default=1000, type=int, help='Number of epochs to reduce LR')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--save_model', default='checkpoint-middy.pt', type=str, help='Filename for saved model')
parser.add_argument('--tolerance', default = 6, type=int, help='Number of epochs before early stoppage')
parser.add_argument('--load_model', default='checkpoint-middy.pt', type=str, help='Load pretrained model')

args=parser.parse_args()

def main():
    device = torch.device("cuda")
    model = graph_model.GCNMid().to(device)
    if args.load_model is not None:
        model.load_state_dict(torch.load(f'./{args.load_model}'))
    model.to(device)
    train_loader, val_loader, test_loader = graph_load.LoadGraphData(data_dir, args.batch_size, args.batch_size)
    model.cuda()
    #optimizer = Adam(model.parameters(), lr = args.lr, weight_decay=args.wd)
    optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = StepLR(optimizer, step_size=args.schedule, gamma=0.95)
    #scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=5, cooldown=5)
    check = 0
    track = 0
    losses = []
    f = open(f'{args.save_model}.txt', 'a')
    for epoch in range(args.epoch):
        print("EPOCH:", epoch)
        f.write("EPOCH:")
        f.write(str(epoch))
        if track < args.tolerance:
            loss = train(train_loader, model, optimizer)
            losses.append(loss.cpu().detach().numpy())
            train_pred, train_target = validate(train_loader, model)
            print(f"Training acc: {accuracy_score(train_target.cpu(), train_pred.cpu())}")
            f.write(f"\nTraining acc: {accuracy_score(train_target.cpu(), train_pred.cpu())}\n")
            pred_val, target_val = validate(val_loader, model)
            print(f'Loss:{loss}')
            f.write(f'Loss:{loss}\n')
            #print(target_val.shape[0])
            #print(sum(target_val))
            #print(sum(pred_val))
            print(f"Val total: {target_val.cpu().shape}, val positive: {target_val.cpu().sum()}")
            print(f"Val acc: {accuracy_score(target_val.cpu(), pred_val.cpu())}")
            print(f"Val F2: {fbeta_score(target_val.cpu(), pred_val.cpu(), beta=2.0)}")
            f.write(f"Val total: {target_val.cpu().shape}, val positive: {target_val.cpu().sum()}\n")
            f.write(f"Val acc: {accuracy_score(target_val.cpu(), pred_val.cpu())}\n")
            f.write(f"Val F2: {fbeta_score(target_val.cpu(), pred_val.cpu(), beta=2.0)}\n")
            check = fbeta_score(target_val.cpu(), pred_val.cpu(), beta=2.0)
            scheduler.step()
            if check < 0.87:
                track = 0
            else:
                track += 1
        elif accuracy_score(target_val.cpu(), pred_val.cpu()) < 0.87:
            track = 0
        else:
            print("Stop condition reached!")
            f.write("Stop condition reached!\n")
            break

    plt.plot(losses)
    plt.savefig(f'{args.save_model}.png')

    torch.save(model.state_dict(), f'./{args.save_model}')    
    pred_test, target_test = test(test_loader, model)
    print(f"Test total: {target_test.cpu().shape}, test positive: {target_test.cpu().sum()}")
    print(f"Test acc: {accuracy_score(target_test.cpu(), pred_test.cpu())}")

    f.write(f"Test total: {target_test.cpu().shape}, test positive: {target_test.cpu().sum()}\n")
    f.write(f"Test acc: {accuracy_score(target_test.cpu(), pred_test.cpu())}\n")
   

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
