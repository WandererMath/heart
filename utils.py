import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
import torch
import torch.nn.functional as F
import torch.nn as nn


EKG_SPLIT_INDEX=299
DIRECTORY='../data/plan1'



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
model3 = TransformerClassifier(input_dim=20, num_classes=2)
model2 = TransformerClassifier(input_dim=29, num_classes=2)
#model.load_state_dict(torch.load(f'model{NTH_PLAN}.pth'))#, map_location=torch.device('cpu')))





def diff(seq):
    return [seq[i]-seq[i-1] for i in range(1, len(seq))]

def find_first_local_max(seq):
    diffed=diff(seq)
    for i in range(len(diffed)-1):
        if diffed[i+1]<0 and diffed[i]>0:
            return i
    return None

def find_max_index(seq):
    L=100
    H=250
    part=seq[L:H]
    m=max(part)
    i=list(part).index(m)
    return i+L

def heart_rate(filename):
    # Load the PNG file
    image = plt.imread(filename)
    # Convert the image to a NumPy array (already handled by plt.imread)
    image_array = np.array(image)

    raw=np.transpose(image_array[EKG_SPLIT_INDEX:])
    A=raw[:400, ]
    B=raw.T
    A=A.T
    #plt.imshow(raw.T)
    #plt.savefig("tmp.png")

    # Convolution along the horizontal axis
    #breakpoint()
    output = []
    for x in range(500):
    #for x in range(B.shape[1] - A.shape[1] + 1):  # Moving A along the horizontal axis
        sub_B = B[:, x:x + A.shape[1]]  # Extract a slice of B with the same width as A
        conv = np.sum(A * sub_B)  # Element-wise multiplication and sum (convolution)
        output.append(conv)

    # Convert output to a numpy array
    output = np.array(output)

    # Plot the result
    '''
    plt.plot(output)
    plt.title("Convolution of A moving along the horizontal axis of B")
    plt.xlabel("Horizontal Position (x)")
    plt.ylabel("Convolution Value")
    plt.savefig("tmp_conv.png")'''
    return find_max_index(output)

def predict_1(sample, model):
    sample=torch.tensor([sample], dtype=torch.float32)
    output=model(sample)
    _, predicted=output.max(1)
    return predicted

if __name__=='__main__':
    a=heart_rate('264_het_12wk_bl_2019-10-18-13-16-27.avisummarygray.png')
    print(a)