import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ----------------------------
# 1. GRAPH CONNECTIVITY
# ----------------------------
def get_adjacency_matrix(num_nodes=543):
    adj = np.eye(num_nodes)
    
    def get_hand_conn(offset):
        return [(offset+i, offset+j) for i, j in [
            (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12), (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20)]]

    if num_nodes == 543:
        # Full Holistic: Pose(0-32), Face(33-500), LH(501-521), RH(522-542)
        pose_conn = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (23, 24)]
        lh_conn = get_hand_conn(501) + [(15, 501)] # Left Wrist to Left Shoulder
        rh_conn = get_hand_conn(522) + [(16, 522)] # Right Wrist to Right Shoulder
        # Strategic connection: Hands to Face (Mouth corners)
        cross_conn = [(508, 61), (529, 291)] 
        connections = pose_conn + lh_conn + rh_conn + cross_conn
    else:
        # Hands-Only (42 nodes)
        connections = get_hand_conn(0) + get_hand_conn(21)

    for i, j in connections:
        if i < num_nodes and j < num_nodes:
            adj[i, j] = adj[j, i] = 1
    
    # Normalized Laplacian
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    adj_normalized = np.diag(d_inv_sqrt).dot(adj).dot(np.diag(d_inv_sqrt))
    return torch.FloatTensor(adj_normalized)

# ----------------------------
# 2. TGCN ARCHITECTURE
# ----------------------------
class GraphConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_ch, out_ch) * 0.01)

    def forward(self, x, adj):
        # x: (B, T, N, C)
        x = torch.einsum("ij,btnc->btic", adj, x)
        return torch.einsum("btic,co->btio", x, self.W)

class TGCN(nn.Module):
    def __init__(self, num_nodes, num_classes, adj):
        super().__init__()
        self.num_nodes = num_nodes
        self.adj = nn.Parameter(adj, requires_grad=False)
        self.gcn = GraphConv(3, 32)
        self.lstm = nn.LSTM(num_nodes * 32, 512, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        B, T, _ = x.shape
        x = x.view(B, T, self.num_nodes, 3)
        x = F.relu(self.gcn(x, self.adj))
        x = x.view(B, T, -1)
        x, _ = self.lstm(x)
        # Take mean of temporal dimension for better global sign understanding
        x = torch.mean(x, dim=1) 
        return self.fc(self.dropout(x))

# ----------------------------
# 3. ROBUST DATASET LOADER
# ----------------------------
class HolisticDataset(Dataset):
    def __init__(self, root, seq_len=48, mode="holistic"):
        self.samples, self.labels = [], []
        self.seq_len = seq_len
        self.mode = mode

        for fname in tqdm(sorted(os.listdir(root)), desc="Parsing Dataset"):
            if not fname.endswith(".json"): continue
            
            with open(os.path.join(root, fname)) as f:
                data = json.load(f)
            
            seq = []
            for frame_data in data:
                # Matches your specific JSON structure
                lm = frame_data.get("landmarks", [])
                if len(lm) == 1629:
                    pts = np.array(lm).reshape(543, 3)
                    if mode == "hands": pts = pts[501:, :]
                    seq.append(pts)
            
            if len(seq) < 10: continue # Quality control

            # Temporal Padding/Truncation
            if len(seq) < seq_len:
                seq += [seq[-1]] * (seq_len - len(seq))
            else:
                seq = seq[:seq_len]
            
            self.samples.append(np.stack(seq))
            # Extract Label (assuming WORD_ID.json format)
            self.labels.append(fname.split("_")[0].upper())

        self.encoder = LabelEncoder()
        self.encoded_labels = self.encoder.fit_transform(self.labels)
        self.classes = self.encoder.classes_.tolist()

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)
        
        # --- SCALE & POSITION NORMALIZATION ---
        if self.mode == "holistic":
            # Anchor to Nose (0)
            anchor = x[:, 0:1, :]
            # Scale by Shoulder Width (indices 11, 12)
            shoulder_width = torch.dist(x[0, 11, :], x[0, 12, :]) + 1e-6
            x = (x - anchor) / shoulder_width
        else:
            # Anchor to Right Wrist (index 21 in hands slice)
            x = x - x[:, 21:22, :]

        return x.view(self.seq_len, -1), torch.tensor(self.encoded_labels[idx])

# ----------------------------
# 4. TRAINING ENGINE
# ----------------------------
def train():
    JSON_DIR = "./holistic_features"
    MODE = "holistic"
    BATCH_SIZE = 32
    EPOCHS = 360
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = HolisticDataset(JSON_DIR, mode=MODE)
    num_classes = len(dataset.classes)
    num_nodes = 543 if MODE == "holistic" else 42
    
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    adj = get_adjacency_matrix(num_nodes).to(DEVICE)
    model = TGCN(num_nodes, num_classes, adj).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    print(f"Dataset Loaded: {len(dataset)} samples, {num_classes} classes.")

    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        # Validation phase
        model.eval()
        v_loss, correct = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                v_loss += criterion(out, y).item()
                correct += (out.argmax(1) == y).sum().item()
        
        avg_v_loss = v_loss/len(val_loader)
        scheduler.step(avg_v_loss)
        
        print(f"Epoch {epoch+1:02d} | Loss: {t_loss/len(train_loader):.4f} | Val Acc: {correct/val_size:.2f}")

    # --- SAVE COMPLETE BUNDLE ---
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': dataset.classes,
        'mode': MODE,
        'num_nodes': num_nodes,
        'seq_len': dataset.seq_len
    }, f"FULL_TGCN_{MODE}.pth")
    print(f"Model saved with {num_classes} classes.")

if __name__ == "__main__":
    train()