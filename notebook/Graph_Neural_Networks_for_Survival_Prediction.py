titanic_path = r"C:\Users\HP\OneDrive\Desktop\Kim Jong UN Confidentials\Projects\Graph Neural Networks for Survival Prediction- A Novel Biostatistical Approach to the Titanic Dataset\notebook\titanic"


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



train_file_path = os.path.join(titanic_path, 'train.csv')
train = pd.read_csv(train_file_path)


most_common_embarked = train['Embarked'].mode()[0]

train['Embarked'] = train['Embarked'].fillna(most_common_embarked)
mean_age = train['Age'].mean()

train['Age'] = train['Age'].fillna(mean_age)

train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
embarked_dummies = pd.get_dummies(train['Embarked'], prefix='Embarked')
train = pd.concat([train, embarked_dummies], axis=1)
train.drop('Embarked', axis=1, inplace=True)
pclass_dummies = pd.get_dummies(train['Pclass'], prefix='Pclass')
train = pd.concat([train, pclass_dummies], axis=1)
train.drop('Pclass', axis=1, inplace=True)


train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

X = train.drop('Survived', axis=1)
y = train['Survived']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

train_idx, val_idx = train_test_split(range(len(X)), test_size=0.2, random_state=42, stratify=y)
print(f"Train set size: {len(train_idx)}, Validation set size: {len(val_idx)}")


distances = pairwise_distances(X_scaled)
k = 5
knn_indices = np.argsort(distances, axis=1)[:, 1:k+1]
edges = []
for i in range(len(X)):
    for j in knn_indices[i]:
        edges.append([i, j])
        edges.append([j, i])
edges = np.unique(np.array(edges), axis=0)



x = torch.tensor(X_scaled, dtype=torch.float)
edge_index = torch.tensor(edges.T, dtype=torch.long)
y_torch = torch.tensor(y.values, dtype=torch.long)
train_mask = torch.zeros(len(X), dtype=torch.bool)
train_mask[train_idx] = True
val_mask = torch.zeros(len(X), dtype=torch.bool)
val_mask[val_idx] = True
data = Data(x=x, edge_index=edge_index, y=y_torch, train_mask=train_mask, val_mask=val_mask)
print(f"Train mask sum: {data.train_mask.sum().item()}, Val mask sum: {data.val_mask.sum().item()}")


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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(input_dim=X_scaled.shape[1], hidden_dim=16, output_dim=2).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


train_losses = []
val_losses = []



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


output_dir = './model_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")



torch.save(model.state_dict(), os.path.join(output_dir, 'titanic_gcn_model.pth'))
torch.save(model, os.path.join(output_dir, 'titanic_gcn_model.pt'))

print(f"Model state dict saved to {os.path.join(output_dir, 'titanic_gcn_model.pth')}")
print(f"Entire model saved to {os.path.join(output_dir, 'titanic_gcn_model.pt')}")

model.eval()
with torch.no_grad():
    val_out = model(data)
    _, pred = val_out.max(dim=1)
    val_pred = pred[data.val_mask].cpu().numpy()
    val_true = data.y[data.val_mask].cpu().numpy()
    val_probs = torch.softmax(val_out[data.val_mask], dim=1)[:, 1].cpu().numpy()
    print(f"val_true shape: {val_true.shape}, val_pred shape: {val_pred.shape}")


f1 = f1_score(val_true, val_pred)
print(f'Validation F1 Score: {f1:.4f}')


try:
    cm = confusion_matrix(val_true, val_pred)
    print(f"Confusion Matrix:\n{cm}")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    plt.show()
    plt.close()
    if os.path.exists(confusion_matrix_path):
        print(f"Confusion matrix saved successfully at {confusion_matrix_path}")
    else:
        print("Failed to save confusion matrix")
except Exception as e:
    print(f"Error in confusion matrix plotting: {str(e)}")

try:
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    loss_curves_path = os.path.join(output_dir, 'loss_curves.png')
    plt.savefig(loss_curves_path)
    plt.show()
    plt.close()
    if os.path.exists(loss_curves_path):
        print(f"Loss curves saved successfully at {loss_curves_path}, Size: {os.path.getsize(loss_curves_path)} bytes")
    else:
        print("Failed to save loss curves")
except Exception as e:
    print(f"Error in loss curve plotting: {str(e)}")

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
    roc_curve_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(roc_curve_path)
    plt.show()
    plt.close()
    if os.path.exists(roc_curve_path):
        print(f"ROC curve saved successfully at {roc_curve_path}, Size: {os.path.getsize(roc_curve_path)} bytes")
    else:
        print("Failed to save ROC curve")
except Exception as e:
    print(f"Error in ROC curve plotting: {str(e)}")




import networkx as nx
import matplotlib.pyplot as plt


G = nx.Graph()
G.add_nodes_from(range(len(y)))
G.add_edges_from(edges)
node_colors = ['green' if label == 1 else 'red' for label in y]


plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, node_color=node_colors, node_size=50, edge_color='gray', with_labels=False)
plt.title('k-NN Graph of Titanic Passengers (k=5, Colored by Survival)')

graph_path = os.path.join(output_dir, 'titanic_knn_graph.png')
plt.savefig(graph_path)
plt.show()
plt.close()

import os
if os.path.exists(graph_path):
    print(f"Graph visualization saved successfully at {graph_path}, Size: {os.path.getsize(graph_path)} bytes")
else:
    print("Failed to save graph visualization")



sub_nodes = range(100)
sub_edges = [edge for edge in edges if edge[0] in sub_nodes and edge[1] in sub_nodes]
G = nx.Graph()
G.add_nodes_from(sub_nodes)
G.add_edges_from(sub_edges)
node_colors = ['green' if y[i] == 1 else 'red' for i in sub_nodes]

plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, node_color=node_colors, node_size=100, edge_color='gray', with_labels=False)
plt.title('k-NN Subgraph of First 100 Titanic Passengers (k=5, Colored by Survival)')

subgraph_path = os.path.join(output_dir, 'titanic_knn_subgraph.png')
plt.savefig(subgraph_path)
plt.show()
plt.close()

if os.path.exists(subgraph_path):
    print(f"Subgraph visualization saved successfully at {subgraph_path}, Size: {os.path.getsize(subgraph_path)} bytes")
else:
    print("Failed to save subgraph visualization")

import networkx as nx
import numpy as np



print("Edges shape:", edges.shape)
print("Sample edges:", edges[:5])
print("Edge types:", [type(e[0]) for e in edges[:5]], [type(e[1]) for e in edges[:5]])
print("Unique node IDs:", np.unique(edges))
print("Node ID types:", [type(n) for n in np.unique(edges)][:5])



G = nx.Graph()
G.add_nodes_from(range(len(y)))
G.add_edges_from(edges)



print("Node types in G:", [type(n) for n in list(G.nodes())][:5])
print("Edge sample in G:", list(G.edges())[:5])



edges = np.unique(np.array(edges), axis=0)

G = nx.Graph()
G.add_nodes_from(range(len(y)))
G.add_edges_from(edges)
node_colors = ['green' if label == 1 else 'red' for label in y]
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, node_color=node_colors, node_size=50, edge_color='gray', with_labels=False)
plt.title('k-NN Graph of Titanic Passengers (k=5, Colored by Survival)')

graph_path = os.path.join(output_dir, 'titanic_knn_graph.png')
plt.savefig(graph_path)
plt.show()
plt.close()
if os.path.exists(graph_path):
    print(f"Graph visualization saved successfully at {graph_path}, Size: {os.path.getsize(graph_path)} bytes")
else:
    print("Failed to save graph visualization")

degrees = [d for n, d in G.degree()]
plt.figure(figsize=(8, 6))
plt.hist(degrees, bins=20, color='blue', alpha=0.7)
plt.title('Node Degree Distribution in k-NN Graph')
plt.xlabel('Degree')
plt.ylabel('Frequency')
degree_distribution_path = os.path.join(output_dir, 'degree_distribution.png')
plt.savefig(degree_distribution_path)
plt.show()
plt.close()

