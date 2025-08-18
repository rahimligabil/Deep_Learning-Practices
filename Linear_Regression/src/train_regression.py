import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data import start_data
from model import build_linear


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total, n = 0.0,0
    for xb,yb in loader:
        xb,yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred,yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)

@torch.no_grad()
def evaluate(model,loader,device):
    model.eval()

    preds,trues = [], []

    for xb,yb in loader:
        xb, yb = xb.to(device),yb.to(device)
        pred = model(xb)
        preds.append(pred.cpu().numpy().ravel())
        trues.append(yb.cpu().numpy().ravel())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    return mse, mae, r2




# ana kısım
def main():
    device = get_device()
    print("Using device:", device)

    # 1. datayı hazırla
    train_loader, val_loader, test_loader, input_dim = start_data()

    # 2. model kur
    model = build_linear(input_dim).to(device)

    # 3. loss + optimizer seç
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # 4. epoch döngüsü
    for epoch in range(50):   # 50 epoch mesela
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_mse, val_mae, val_r2 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1:03d}: Train Loss={train_loss:.4f} | Val MSE={val_mse:.4f} | Val MAE={val_mae:.4f} | Val R2={val_r2:.4f}")

    # 5. en sonda test seti
    test_mse, test_mae, test_r2 = evaluate(model, test_loader, device)
    print("\nTest Results:")
    print(f"MSE={test_mse:.4f}, MAE={test_mae:.4f}, R2={test_r2:.4f}")


if __name__ == "__main__":
    main()
