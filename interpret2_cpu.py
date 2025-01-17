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

NTH_PLAN=2

with open(f"plan{NTH_PLAN}.pkl", 'rb') as f:
    data=pickle.load(f)
    labels=pickle.load(f)

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

n_sample, n_time, n_feature=data.shape

# Initialize model
model = TransformerClassifier(input_dim=n_feature, num_classes=2)
model.load_state_dict(torch.load(f'model{NTH_PLAN}.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

def main(i):
    input_tensor=np.array([data[i]])
    input_tensor = torch.from_numpy(input_tensor).float()
    input_tensor.requires_grad = True

    outputs = model(input_tensor)
    target_class = torch.argmax(outputs, dim=1)
    outputs[0, target_class].backward()
    gradients = input_tensor.grad[0].detach().numpy()
    gradient_importance = np.abs(gradients)
    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(gradient_importance.T, aspect='auto', cmap='hot', interpolation='nearest')
    plt.colorbar(label='Feature Importance')
    plt.title('Input Feature Importance Heatmap')
    plt.xlabel('Time Step')
    plt.ylabel('Features')
    plt.savefig(f"examples/{i}.pdf")

for k in range(5):
    main(randint(0, n_sample-1))