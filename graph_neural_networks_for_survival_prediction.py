import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
import os

# Load dataset
train = pd.read_csv('/kaggle/input/titanic/train.csv')

# Handle missing values
most_common_embarked = train['Embarked'].mode()[0]
train['Embarked'] = train['Embarked'].fillna(most_common_embarked)
mean_age = train['Age'].mean()
train['Age'] = train['Age'].fillna(mean_age)

# Encode categorical variables
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
embarked_dummies = pd.get_dummies(train['Embarked'], prefix='Embarked')
train = pd.concat([train, embarked_dummies], axis=1)
train.drop('Embarked', axis=1, inplace=True)
pclass_dummies = pd.get_dummies(train['Pclass'], prefix='Pclass')
train = pd.concat([train, pclass_dummies], axis=1)
train.drop('Pclass', axis=1, inplace=True)

# Drop irrelevant columns
train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Define features and target
X = train.drop('Survived', axis=1)
y = train['Survived']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split train and validation sets
train_idx, val_idx = train_test_split(range(len(X)), test_size=0.2, random_state=42, stratify=y)
print(f"Train set size: {len(train_idx)}, Validation set size: {len(val_idx)}")

# Construct the graph
distances = pairwise_distances(X_scaled)
k = 5
knn_indices = np.argsort(distances, axis=1)[:, 1:k+1]
edges = []
for i in range(len(X)):
    for j in knn_indices[i]:
        edges.append([i, j])
        edges.append([j, i])
edges = np.unique(np.array(edges), axis=0)

# Create PyTorch Geometric Data object
x = torch.tensor(X_scaled, dtype=torch.float)
edge_index = torch.tensor(edges.T, dtype=torch.long)
y_torch = torch.tensor(y.values, dtype=torch.long)
train_mask = torch.zeros(len(X), dtype=torch.bool)
train_mask[train_idx] = True
val_mask = torch.zeros(len(X), dtype=torch.bool)
val_mask[val_idx] = True
data = Data(x=x, edge_index=edge_index, y=y_torch, train_mask=train_mask, val_mask=val_mask)
print(f"Train mask sum: {data.train_mask.sum().item()}, Val mask sum: {data.val_mask.sum().item()}")

# Define GCN model
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize model and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(input_dim=X_scaled.shape[1], hidden_dim=16, output_dim=2).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Lists to store losses
train_losses = []
val_losses = []

# Training loop
num_epochs = 100
val_block_count = 0
val_append_count = 0
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    try:
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    except Exception as e:
        print(f"Training error in epoch {epoch+1}: {str(e)}")
        continue
    
    # Validation
    model.eval()
    with torch.no_grad():
        if data.val_mask.sum() == 0:
            raise ValueError("Validation mask is empty. Check train_test_split or val_mask initialization.")
        try:
            val_out = model(data)
            print(f"Epoch {epoch+1}, val_out shape: {val_out.shape}, val_mask shape: {data.val_mask.shape}, y shape: {data.y.shape}")
            if val_out[data.val_mask].shape[0] != data.y[data.val_mask].shape[0]:
                raise ValueError(f"Shape mismatch: val_out[data.val_mask] {val_out[data.val_mask].shape}, y[data.val_mask] {data.y[data.val_mask].shape}")
            val_loss = F.nll_loss(val_out[data.val_mask], data.y[data.val_mask])
            val_losses.append(val_loss.item())
            val_append_count += 1
            val_block_count += 1
            print(f"Epoch {epoch+1}, Val Loss: {val_loss.item():.4f}, Val Losses Length: {len(val_losses)}, Val Append Count: {val_append_count}")
            _, pred = val_out.max(dim=1)
            correct = pred[data.val_mask].eq(data.y[data.val_mask]).sum().item()
            acc = correct / data.val_mask.sum().item() if data.val_mask.sum().item() > 0 else 0.0
            print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {acc:.4f}")
        except Exception as e:
            print(f"Validation error in epoch {epoch+1}: {str(e)}")
            val_block_count += 1
            continue

print(f"Validation block executed {val_block_count} times")
print(f"Length of train_losses: {len(train_losses)}, Length of val_losses: {len(val_losses)}")

# Save model
torch.save(model.state_dict(), '/kaggle/working/titanic_gcn_model.pth')
torch.save(model, '/kaggle/working/titanic_gcn_model.pt')

# Visualization and Metrics
model.eval()
with torch.no_grad():
    val_out = model(data)
    _, pred = val_out.max(dim=1)
    val_pred = pred[data.val_mask].cpu().numpy()
    val_true = data.y[data.val_mask].cpu().numpy()
    val_probs = torch.softmax(val_out[data.val_mask], dim=1)[:, 1].cpu().numpy()
    print(f"val_true shape: {val_true.shape}, val_pred shape: {val_pred.shape}")

# F1 Score
f1 = f1_score(val_true, val_pred)
print(f'Validation F1 Score: {f1:.4f}')

# Confusion Matrix
try:
    cm = confusion_matrix(val_true, val_pred)
    print(f"Confusion Matrix:\n{cm}")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('/kaggle/working/confusion_matrix.png')
    plt.show()
    plt.close()
    if os.path.exists('/kaggle/working/confusion_matrix.png'):
        print("Confusion matrix saved successfully at /kaggle/working/confusion_matrix.png")
    else:
        print("Failed to save confusion matrix")
except Exception as e:
    print(f"Error in confusion matrix plotting: {str(e)}")

# Training and Validation Loss Curves
try:
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('/kaggle/working/loss_curves.png')
    plt.show()
    plt.close()
    if os.path.exists('/kaggle/working/loss_curves.png'):
        print(f"Loss curves saved successfully at /kaggle/working/loss_curves.png, Size: {os.path.getsize('/kaggle/working/loss_curves.png')} bytes")
    else:
        print("Failed to save loss curves")
except Exception as e:
    print(f"Error in loss curve plotting: {str(e)}")

# ROC Curve
try:
    fpr, tpr, _ = roc_curve(val_true, val_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('/kaggle/working/roc_curve.png')
    plt.show()
    plt.close()
    if os.path.exists('/kaggle/working/roc_curve.png'):
        print(f"ROC curve saved successfully at /kaggle/working/roc_curve.png, Size: {os.path.getsize('/kaggle/working/roc_curve.png')} bytes")
    else:
        print("Failed to save ROC curve")
except Exception as e:
    print(f"Error in ROC curve plotting: {str(e)}")
