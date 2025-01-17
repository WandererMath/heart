import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchviz import make_dot
import numpy as np
import matplotlib.pyplot as plt



import pickle

NTH_PLAN=2

# Custom Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)  # Labels should remain as class indices (0, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Transformer Classifier Model
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


# Prepare Dummy Data
def generate_dummy_data(num_samples=1000, seq_len=50, input_dim=10):
    data = np.random.rand(num_samples, seq_len, input_dim)
    labels = np.random.randint(0, 2, size=num_samples)  # Class indices (0 or 1)
    return data, labels


# Get device for training
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Main
#data, labels = generate_dummy_data()
with open(f"plan{NTH_PLAN}.pkl", 'rb') as f:
    data=pickle.load(f)
    data=np.array(data)
    labels=pickle.load(f)

dataset = TimeSeriesDataset(data, labels)
# Split Data
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

print("Data loaded")

# Training Parameters
input_dim = data.shape[2]
num_classes = 2
model = TransformerClassifier(input_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training Loop
num_epochs = 50
train_losses = []
val_losses = []
val_accuracies = []
train_accuracies=[]
print("Start training")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_total =0
    train_correct=0 
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets).sum().item()
    train_loss /= len(train_loader)
    train_acc = 100. * train_correct / train_total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    val_loss /= len(val_loader)
    val_acc = 100. * correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

print("Done")
torch.save(model.state_dict(), f"model{NTH_PLAN}.pth")
print("Model Saved")


# Plotting
plt.figure(figsize=(12, 6))

# Plotting Training and Validation Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plotting Training and Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(f"Accuracy{NTH_PLAN}.pdf")

