import os
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

class FrameDataset(Dataset):
    def __init__(self, root_dir):
        self.paths = []
        for video in sorted(os.listdir(root_dir)):
            video_path = os.path.join(root_dir, video)
            for frame in sorted(os.listdir(video_path)):
                self.paths.append(os.path.join(video_path, frame))

        self.transform = transforms.Compose([
            transforms.ToTensor(),   # [0,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = self.transform(img)
        return img
    
class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

train_dataset = FrameDataset(r"D:\pytorch\Avenue_Corrupted-20251221T112159Z-3-001\Avenue_Corrupted\Dataset\training_videos")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = ConvAutoEncoder().to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for imgs in tqdm(train_loader):
        imgs = imgs.to(DEVICE)

        recon = model(imgs)
        loss = criterion(recon, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.6f}")

model.eval()
results = []

test_root = r"D:\pytorch\Avenue_Corrupted-20251221T112159Z-3-001\Avenue_Corrupted\Dataset\testing_videos"

with torch.inference_mode():
    for video in sorted(os.listdir(test_root)):
        video_path = os.path.join(test_root, video)
        frames = sorted(os.listdir(video_path))

        for frame in frames:
            frame_path = os.path.join(video_path, frame)

            img = cv2.imread(frame_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = torch.tensor(img / 255.0).permute(2,0,1).unsqueeze(0).float().to(DEVICE)

            recon = model(img)
            error = torch.mean((recon - img) ** 2).item()

            frame_num = int(frame.split('.')[0].split('_')[-1])
            frame_id = f"{int(video)}_{frame_num}"

            results.append([frame_id, error])
df = pd.DataFrame(results, columns=["Id", "Predicted"])
# Normalize scores to [0,1]
df["Predicted"] = (df["Predicted"] - df["Predicted"].min()) / \
                  (df["Predicted"].max() - df["Predicted"].min())

df.to_csv("submission.csv", index=False)
df.head()