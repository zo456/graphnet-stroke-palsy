import graph_load
import graph_model
import torch
import argparse
from sklearn.metrics import accuracy_score, recall_score, precision_score, fbeta_score, f1_score

device = 'cuda'
#data_dir = '../celeba/'
data_dir = '../guoetal-arraydata-tnf-pr/Data/'

parser = argparse.ArgumentParser(description='Testing arguments')
parser.add_argument('--load_model', default='3-13-t1.pt', type=str, help='Saved model')

args=parser.parse_args()

def main():
    model = graph_model.GCNMid().to(device)
    test_loader = graph_load.TestData(data_dir, 128)
    model.cuda()
    model.load_state_dict(torch.load(f'./{args.load_model}'))
    pred_test, target_test = test(test_loader, model)

    wrong_pos = (target_test * (target_test != pred_test)).sum()
    wrong_pred = ((target_test != pred_test)).sum()
    
    print(f"Test acc: {accuracy_score(target_test.cpu(), pred_test.cpu())}")
    print(f"Test prec: {precision_score(target_test.cpu(), pred_test.cpu())}")
    print(f"Test rec: {recall_score(target_test.cpu(), pred_test.cpu())}")
    print(f"Test F1: {f1_score(target_test.cpu(), pred_test.cpu())}")  
    print(f"Test F2: {fbeta_score(target_test.cpu(), pred_test.cpu(), beta=2.0)}")
    print(f"Total: {target_test.cpu().shape}, total positive: {target_test.cpu().sum()}")
    print(f"Wrongly predicted: {wrong_pred}, wrongly predicted positive: {wrong_pos}") 
    
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
