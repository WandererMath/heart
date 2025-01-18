import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchviz import make_dot
from torch.autograd import Function
import numpy as np
import matplotlib.pyplot as plt

import pickle
from random import randint

NTH_PLAN=3

with open(f"plan{NTH_PLAN}.pkl", 'rb') as f:
    pickle.load(f)
    data_dict=pickle.load(f)
    labels_dict=pickle.load(f)

print("Data loaded")



class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, 500, d_model))  # Max sequence length of 500
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # Classification token

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = self.embedding(x)
        x += self.positional_encoding[:, :seq_len, :]  # Add positional encoding
        cls_token = self.cls_token.repeat(batch_size, 1, 1)  # Add classification token
        x = torch.cat([cls_token, x], dim=1)
        x = self.encoder(x)  # (batch_size, seq_len, d_model)
        out = self.fc(x[:, 0, :])  # Use classification token output
        return out

# Initialize model
model = TransformerClassifier(input_dim=20, num_classes=2)
model.load_state_dict(torch.load(f'model{NTH_PLAN}.pth'))#, map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# returns percentage of diseased for  (HET, db)
def predict(data, labels):
    s=[0,0]
    total=[0,0]
    for sample, l in zip(data, labels):
        sample=torch.tensor([sample], dtype=torch.float32)
        output=model(sample)

        _, predicted=output.max(1)
        total[l]+=1
        if predicted==1:
            s[l]+=1
    return s[0]/total[0], s[1]/total[1]

with open("test3.txt", 'w') as f:
    for wk in [12, 16, 36]:
        s0, s1=predict(data_dict[wk], labels_dict[wk])
        f.write(f"wk{wk}\t{s0}\t{s1}\n") 