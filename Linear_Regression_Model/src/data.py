import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM_COLS = ["age", "bmi", "children"]         
CAT_COLS = ["sex", "smoker", "region"]        


def build_preprocessor():

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CAT_COLS),
        ],
        remainder="drop"
    )
    return pre

def start_data(batch_size: int = 64, seed: int = 42, scale_target: bool = False):

    df = pd.read_csv("src/data/insurance.csv")
    print(df.head())  

    y = df["charges"].values.astype(np.float32)     
    X = df.drop(columns=["charges"])

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=seed
    )
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=seed
    )

    pre = build_preprocessor()
    X_tr_np = pre.fit_transform(X_tr)   
    X_va_np = pre.transform(X_va)
    X_te_np = pre.transform(X_te)

   
    X_tr_t = torch.from_numpy(X_tr_np.astype(np.float32))
    X_va_t = torch.from_numpy(X_va_np.astype(np.float32))
    X_te_t = torch.from_numpy(X_te_np.astype(np.float32))

    Y_tr_t = torch.from_numpy(y_tr).unsqueeze(1)  # (N,1)
    Y_va_t = torch.from_numpy(y_va).unsqueeze(1)
    Y_te_t = torch.from_numpy(y_te).unsqueeze(1)

 
    train_loader = DataLoader(
        TensorDataset(X_tr_t, Y_tr_t),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_va_t, Y_va_t),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(X_te_t, Y_te_t),
        batch_size=batch_size,
        shuffle=False,
    )

    input_dim = X_tr_t.shape[1]

    return train_loader, val_loader, test_loader, input_dim
