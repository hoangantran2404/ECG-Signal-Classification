#Library
import torch
import os
import torch.nn as nn
import numpy as np
import pandas as pd
import tqdm
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

MODEL_PATH = Path(os.path.expanduser("~/Documents/CNN_DNN/project/project1/model"))
MODEL_NAME = "ecg_tcn_cnn.pth" 
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

data_path = Path(os.path.expanduser("~/Documents/CNN_DNN/project/project1/data"))
train_dir = data_path/"mitbih_train.csv"
test_dir = data_path/"mitbih_test.csv"

class Transform_to_Dataset(Dataset):
    def __init__(self, csv_file, transform=False):
        self.data = pd.read_csv(csv_file, header=None)
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        row = self.data.iloc[index]
        sample = row[:-1].values.astype(np.float32)
        label = int(row.iloc[-1])

        if self.transform:
            if np.random.rand() > 0.5:
                scale_factor = np.random.uniform(0.8, 1.2)
                sample = sample * scale_factor

            if np.random.rand() > 0.5:
                noise = np.random.normal(0, 0.02, sample.shape)
                sample = sample + noise

            if np.random.rand() > 0.5:
                shift = np.random.randint(-15, 15)
                sample = np.roll(sample, shift)
                if shift > 0:
                    sample[:shift] = 0
                else:
                    sample[shift:] = 0

            if np.random.rand() > 0.7: 
                mask_len = np.random.randint(5, 25)
                mask_start = np.random.randint(0, len(sample) - mask_len)
                sample[mask_start : mask_start + mask_len] = 0
            
            if np.random.rand() > 0.7:
                slope = np.random.uniform(-0.1, 0.1)
                drift = np.linspace(0, slope, len(sample))
                sample = sample + drift

        sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return sample_tensor, label_tensor

# Setup DataLoader
train_dataset = Transform_to_Dataset(train_dir, transform=True)
test_dataset = Transform_to_Dataset(test_dir, transform=False)
NUM_WORKERS = min(os.cpu_count(), 4)

train_dataloader = DataLoader(dataset=train_dataset, 
                              batch_size=32, 
                              shuffle=True, 
                              num_workers=NUM_WORKERS, 
                              pin_memory=True)

test_dataloader = DataLoader(dataset=test_dataset, 
                             batch_size=32, 
                             shuffle=False, 
                             num_workers=NUM_WORKERS, 
                             pin_memory=True)



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_shape, out_shape, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = in_shape, 
                               out_channels = out_shape, 
                               kernel_size=kernel_size, 
                               stride=stride, padding=padding, 
                               dilation=dilation, 
                               bias=False)
        self.bn1 = nn.BatchNorm1d(out_shape)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(in_channels = out_shape, 
                               out_channels = out_shape, 
                               kernel_size=kernel_size, 
                               stride=stride, padding=padding, 
                               dilation=dilation, 
                               bias=False)
        self.bn2 = nn.BatchNorm1d(out_shape)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.bn1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.bn2, self.chomp2, self.relu2, self.dropout2)
        if in_shape != out_shape:
            self.reduction_sample_size = nn.Conv1d(in_shape, out_shape, 1)
        else:
            None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        if self.reduction_sample_size is None:
            residual = x 
        else:
            residual = self.reduction_sample_size(x)
        return self.relu(out + residual)


class ECG_TCN_Explicit(nn.Module):
    def __init__(self, num_inputs, num_classes, kernel_size=5, dropout=0.1):  
        super().__init__()
        
        self.layer1 = TemporalBlock(in_shape=num_inputs, out_shape=64, kernel_size=kernel_size, stride=1, 
                                    dilation=1, padding=(kernel_size-1)*1, dropout=dropout)
        
        self.layer2 = TemporalBlock(in_shape=64, out_shape=128, kernel_size=kernel_size, stride=1, 
                                    dilation=2, padding=(kernel_size-1)*2, dropout=dropout)
        
        self.layer3 = TemporalBlock(in_shape=128, out_shape=256, kernel_size=kernel_size, stride=1, 
                                    dilation=4, padding=(kernel_size-1)*4, dropout=dropout)
        
        self.layer4 = TemporalBlock(in_shape=256, out_shape=512, kernel_size=kernel_size, stride=1, 
                                    dilation=8, padding=(kernel_size-1)*8, dropout=dropout)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_avg_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


def accuracy_fn(y_pred, y):
    accuracy = torch.eq(y, y_pred).sum().item()
    percent = (accuracy / len(y)) * 100
    return percent

def train_step(model, dataloader, loss_fn, optimizer, accuracy_fn, device):
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        train_logit = model(X)
        loss = loss_fn(train_logit, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
        train_pred_class = torch.argmax(torch.softmax(train_logit, dim=1), dim=1)
        train_acc += accuracy_fn(y_pred=train_pred_class, y=y)
    train_loss = train_loss/len(dataloader)
    train_acc = train_acc/len(dataloader)
    return train_loss, train_acc
     
def test_step(model,dataloader,loss_fn,accuracy_fn,device):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_logit = model(X)
            loss = loss_fn(test_logit, y)
            test_loss += loss.item()
            test_pred_class = torch.argmax(torch.softmax(test_logit, dim=1), dim=1)
            test_acc += accuracy_fn(y_pred=test_pred_class, y=y)
        test_loss = test_loss/len(dataloader)
        test_acc = test_acc/len(dataloader)
    return test_loss,test_acc


def train_process(model, train_loader, test_loader, loss_fn, optimizer, accuracy_fn, device, epochs, scheduler=None):
    results = {"train_loss":[], "train_acc":[], "test_loss":[], "test_acc":[]}
    
    best_acc = 0.0 
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer, accuracy_fn, device)
        test_loss, test_acc = test_step(model, test_loader, loss_fn, accuracy_fn, device)
        
        if scheduler:
            scheduler.step(test_loss)
            
        print(f"Epoch: {epoch+1} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            print(f"--> Found New Best Model! Acc: {best_acc:.2f}%. Saving...")
            if not MODEL_PATH.exists():
                MODEL_PATH.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    
    print(f"Training Done. Best Accuracy reached: {best_acc:.2f}%")
    return results


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

model_tcn = ECG_TCN_Explicit(num_inputs=1, num_classes=5, kernel_size=5, dropout=0.1).to(device)

if MODEL_SAVE_PATH.exists():
    print(f"==================================================")
    print(f"Found existing model: {MODEL_SAVE_PATH}")
    print(f"Loading weights to continue training...")
    try:
        model_tcn.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print(">>> SUCCESS: Weights loaded!")
    except Exception as e:
        print(f">>> WARNING: Cannot load weights (Architecture changed?). Error: {e}")
        print(">>> Training from scratch.")
    print(f"==================================================")
else:
    print("No existing model found. Training from scratch.")


class_weight = torch.tensor([1.5, 8.0, 3.0, 27.0, 3.0]).to(device) 
loss_fn = nn.CrossEntropyLoss(weight=class_weight)

optimizer = torch.optim.Adam(model_tcn.parameters(), lr=0.0005, weight_decay=1e-5) 

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)

# Train
print("Start Training TCN Pro...")
model_tcn_result = train_process(model_tcn, train_dataloader, test_dataloader, loss_fn, optimizer, accuracy_fn, device, epochs=20, scheduler=scheduler)

# --- PLOT ---
def plot_results(results):
    epochs = range(len(results['train_loss']))
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, results['train_loss'], label='Train Loss')
    plt.plot(epochs, results['test_loss'], label='Test Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, results['train_acc'], label='Train Acc')
    plt.plot(epochs, results['test_acc'], label='Test Acc')
    plt.title('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig("ECG_TCN_Result.png")
    
def plot_confusion_matrix(model, dataloader):
    y_preds = []
    y_trues = []
    class_names = ['N', 'S', 'V', 'F', 'Q']
    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logit = model(X)
            pred = torch.argmax(torch.softmax(logit, dim=1), dim=1)
            y_preds.extend(pred.cpu().numpy())
            y_trues.extend(y.cpu().numpy())   
    cm = confusion_matrix(y_trues, y_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

plot_results(model_tcn_result)
model_tcn.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
plot_confusion_matrix(model_tcn, test_dataloader) 
plt.savefig("ECG_matrix.png")

