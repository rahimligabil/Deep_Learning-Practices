import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report
from model import MyModel


train_images = np.load("src/data/train_images.npy")
train_labels = np.load("src/data/train_labels.npy")
test_images  = np.load("src/data/test_images.npy")
test_labels  = np.load("src/data/test_labels.npy")


def changeNumpy():
    X_train = torch.from_numpy(train_images).float() / 255.0
    Y_train = torch.from_numpy(train_labels).long()
    X_test = torch.from_numpy(test_images).float() / 255.0
    Y_test = torch.from_numpy(test_labels).long()

    return X_train,Y_train,X_test,Y_test


def getTensorDataset(X,Y):
    dataset = TensorDataset(X,Y)
    return dataset 


def getDataLoader(dataset,batch,shuffle1 = True):
    if shuffle1:
        dataloader = DataLoader(dataset,batch_size = batch,shuffle = True)
    else:
        dataloader = DataLoader(dataset,batch_size = batch,shuffle = False)

    return dataloader


def doFlatten(X):
    flatten = nn.Flatten()
    return flatten(X)



def CrossEntropy(X,labels):
    loss = nn.CrossEntropyLoss()
    loss_fn = loss(X,labels)
    return loss_fn


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()  
    
    total_loss = 0.0
    total_correct = 0
    count = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        output = model(xb)             
        loss = loss_fn(output, yb)     
        loss.backward()               
        optimizer.step()               

        total_loss += loss.item() * xb.size(0)  
        _, preds = output.max(1)       
        total_correct += (preds == yb).sum().item()
        count += xb.size(0)

    avg_loss = total_loss / count
    avg_acc  = total_correct / count

    return avg_loss, avg_acc

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    trues, preds_all = [], []   
    correct, total = 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        output = model(xb)
        _, preds = output.max(dim=1)  
        correct += (preds == yb).sum().item()
        total += yb.size(0)

        trues.extend(yb.cpu().tolist())         
        preds_all.extend(preds.cpu().tolist())   

    acc = correct / total
    report = classification_report(trues, preds_all, digits=4)
    return acc, report






def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train,Y_train,X_test,Y_test = changeNumpy()

    train_dataset = getTensorDataset(X_train,Y_train)
    test_dataset = getTensorDataset(X_test,Y_test)

    train_dataloader = getDataLoader(train_dataset,64)
    test_dataloader = getDataLoader(test_dataset,64,False)


    model = MyModel().to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = 1e-3)

    num_epochs = 10

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_dataloader, loss, optimizer, device)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f}")

    
    test_acc, report = evaluate(model, test_dataloader, device)
    print(f"\nTest Accuracy: {test_acc:.4f}\n")
    print(report)

if __name__ == "__main__":
    main()



   


   
        
