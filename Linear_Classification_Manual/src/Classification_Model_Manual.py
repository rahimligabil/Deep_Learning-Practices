import torch
import torch.nn as nn
import struct
import numpy as np
import matplotlib.pyplot as plt

train_images = np.load("src/data/train_images.npy")
train_labels = np.load("src/data/train_labels.npy")
test_images  = np.load("src/data/test_images.npy")
test_labels  = np.load("src/data/test_labels.npy")

def iterate_minibatches(X, y, batch_size, shuffle=True):
    N = X.shape[0]
    indices = torch.arange(N)
    if shuffle:
        indices = indices[torch.randperm(N)]
    for start in range(0, N, batch_size):
        idx = indices[start:start+batch_size]
        yield X[idx], y[idx]


def doFlatten():

    flatten = nn.Flatten()

    flatten_train_images = flatten(torch.from_numpy(train_images).float())

    flatten_test_images = flatten(torch.from_numpy(test_images).float())

    return flatten_train_images,flatten_test_images



def getParametrs(train_images):

    W = torch.from_numpy(np.random.randn(train_images.shape[1],10) * 0.01).float()

    b = torch.from_numpy(np.zeros(10)).float()

    return W,b



def forward(X,W,b):

    return torch.matmul(X,W) + b


def softmax(logits):

    logits = logits - torch.max(logits,dim=1,keepdim = True).values

    s = torch.exp(logits)

    sum = s.sum(dim=1,keepdim = True)

    return s / sum


def cross_entropy(probs,labels):

    N = probs.shape[0]

    correct_probs = probs[torch.arange(N),labels]

    loss = -torch.mean(torch.log(correct_probs))
    return loss


def backward(X, probs, labels):
    N, C = probs.shape

    one_hot = torch.zeros((N, C))
    one_hot[torch.arange(N), labels] = 1

    dlogits = (probs - one_hot) / N  

    dW = X.T @ dlogits             
    db = dlogits.sum(dim=0)           

    return dW, db



def train(X,y,W,b,lr = 0.01,):

    for i in range(100):
        logits = forward(X, W, b)
        probs = softmax(logits)                
        loss = cross_entropy(probs, y)         
        dW, db = backward(X, probs, y)         

    W -= lr * dW
    b -= lr * db

    preds = probs.argmax(dim=1)
    acc = (preds == y).float().mean()
    print(f"Iter {i}, Loss={loss:.4f}, Acc={acc:.4f}")

    return W,b


def main():
    
    X_train, X_test = doFlatten()                
    y_train = torch.from_numpy(train_labels).long()  
    y_test  = torch.from_numpy(test_labels).long()


    W, b = getParametrs(X_train)


    W, b = train(X_train, y_train, W, b, lr=0.1)

    with torch.no_grad():
        probs_test = softmax(forward(X_test, W, b))
        preds_test = probs_test.argmax(dim=1)
        test_acc = (preds_test == y_test).float().mean().item()
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()