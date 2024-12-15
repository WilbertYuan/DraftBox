import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from rich import print
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[bold green]Running on {device}.")

X_train = np.array(pd.read_csv('data/experiment_4/hand-writing/csvTrainImages 60k x 784.csv'))
y_train = np.array(pd.read_csv('data/experiment_4/hand-writing/csvTrainLabel 60k x 1.csv'))
X_test = np.array(pd.read_csv('data/experiment_4/hand-writing/csvTestImages 10k x 784.csv'))
y_test = np.array(pd.read_csv('data/experiment_4/hand-writing/csvTestLabel 10k x 1.csv'))

X_train = X_train.reshape(-1, 1, 28, 28).astype('float32')
X_test = X_test.reshape(-1, 1, 28, 28).astype('float32')

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, 28 * 28)).reshape(-1, 1, 28, 28)
X_test = scaler.transform(X_test.reshape(-1, 28 * 28)).reshape(-1, 1, 28, 28)

X_train = torch.tensor(X_train).to(device)
y_train = torch.tensor(y_train).to(device)
X_test = torch.tensor(X_test).to(device)
y_test = torch.tensor(y_test).to(device)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

batch_size = 196
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv6 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv8 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.conv10 = nn.Conv2d(128, 512, kernel_size=5, padding=2)
        self.conv11 = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(512 * 1 * 1, 128) 
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 512 * 1 * 1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

class CNNModel2(nn.Module):
    def __init__(self):
        super(CNNModel2, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv6 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=5, padding=2)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=5, padding=2)
        self.conv9 = nn.Conv2d(256, 512, kernel_size=5, padding=2)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        self.conv11 = nn.Conv2d(512, 2048, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(batch_size * 4 * 1 * 1, batch_size)
        self.fc2 = nn.Linear(batch_size, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = F.relu(self.conv5(x))
        # x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, 2)
        # x = F.relu(self.conv7(x))
        # x = F.relu(self.conv8(x))
        # x = F.max_pool2d(x, 2)
        # x = F.relu(self.conv9(x))
        # x = F.relu(self.conv10(x))
        # x = F.relu(self.conv11(x))
        # x = F.max_pool2d(x, 2)
        x = x.view(-1, batch_size* 4 * 1 * 1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



model = CNNModel2().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 100
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
best_test_acc = 0

for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0
    epoch_train_correct = 0
    epoch_train_total = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets.squeeze())
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
        _, preds = torch.max(output, 1)
        epoch_train_correct += (preds == targets.squeeze()).sum().item()
        epoch_train_total += targets.size(0)


    train_loss = epoch_train_loss / len(train_loader)
    train_acc = epoch_train_correct / epoch_train_total

    model.eval()
    epoch_val_loss = 0
    epoch_val_correct = 0
    epoch_val_total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            output = model(inputs)
            val_loss = criterion(output, targets.squeeze())
            epoch_val_loss += val_loss.item()
            _, val_preds = torch.max(output, 1)
            epoch_val_correct += (val_preds == targets.squeeze()).sum().item()
            epoch_val_total += targets.size(0)

    val_loss = epoch_val_loss / len(val_loader)
    val_acc = epoch_val_correct / epoch_val_total

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch [{epoch+1}/{epochs}], "
          f"Train Loss: {train_loss:.7f}, Train Acc: {train_acc:.7f}, "
          f"Val Loss: {val_loss:.7f}, Val Acc: {val_acc:.7f}")

    model.eval()
    with torch.no_grad():
        output = model(X_test)
        # print(X_test.shape)
        _, preds = torch.max(output, 1)
        test_acc = (preds == y_test.squeeze()).float().mean()
        print(f"Test Accuracy: {test_acc:.4f}")
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        # torch.save(model.state_dict(), 'result/best_model_cnn2.pth')
    torch.cuda.empty_cache()
print(f"Best Test Accuracy: {best_test_acc:.4f}")

plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png')

plt.figure()
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_curve.png')
