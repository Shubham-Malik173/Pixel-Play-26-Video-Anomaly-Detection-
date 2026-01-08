import os
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 8
K = 4  # number of past frames to predict next
class FutureFrameDataset(Dataset):
    def __init__(self, root_dir, k=4):
        self.samples = []
        self.k = k

        for vid in sorted(os.listdir(root_dir)):
            vpath = os.path.join(root_dir, vid)
            frames = sorted(os.listdir(vpath))

            for i in range(len(frames) - k):
                seq = [os.path.join(vpath, f) for f in frames[i:i+k+1]]
                self.samples.append(seq)

    def __len__(self):
        return len(self.samples)

    def read_img(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.tensor(img / 255.0).permute(2,0,1).float()

    def __getitem__(self, idx):
        paths = self.samples[idx]
        past = torch.cat([self.read_img(p) for p in paths[:-1]], dim=0)
        future = self.read_img(paths[-1])
        return past, future
class FuturePredictor(nn.Module):
    def __init__(self, k):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3*k, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
train_root = r"D:\pytorch\Avenue_Corrupted-20251221T112159Z-3-001\Avenue_Corrupted\Dataset\training_videos"

train_ds = FutureFrameDataset(train_root, K)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

model = FuturePredictor(K).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(EPOCHS):
    model.train()
    running = 0

    for x, y in tqdm(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running/len(train_loader):.5f}")
test_root = r"D:\pytorch\Avenue_Corrupted-20251221T112159Z-3-001\Avenue_Corrupted\Dataset\testing_videos"

model.eval()
results = []
with torch.no_grad():
    for vid in sorted(os.listdir(test_root)):
        vpath = os.path.join(test_root, vid)
        frames = sorted(os.listdir(vpath))

        imgs = []
        for f in frames:
            img = cv2.imread(os.path.join(vpath, f))
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(torch.tensor(img/255.).permute(2,0,1).float())

        for i in range(len(imgs)):
            frame_num = int(frames[i].split('_')[-1].split('.')[0])

            if i < K:
                results.append([f"{int(vid)}_{frame_num}", 0.0])
                continue

            past = torch.cat(imgs[i-K:i], dim=0).unsqueeze(0).to(DEVICE)
            future = imgs[i].unsqueeze(0).to(DEVICE)

            pred = model(past)
            error = torch.mean((pred - future)**2).item()

            results.append([f"{int(vid)}_{frame_num}", error])
df = pd.DataFrame(results, columns=["Id", "Predicted"])

df[["vid","frame"]] = df["Id"].str.split("_", expand=True).astype(int)
df = df.sort_values(["vid","frame"])

df["Predicted"] = (
    df.groupby("vid")["Predicted"]
      .rolling(window=7, center=True, min_periods=1)
      .mean()
      .reset_index(level=0, drop=True)
)

df["Predicted"] = (df["Predicted"] - df["Predicted"].min()) / \
                  (df["Predicted"].max() - df["Predicted"].min())
df[["Id","Predicted"]].to_csv(
    "submission_future.csv",
    index=False
)

print(df.head())
