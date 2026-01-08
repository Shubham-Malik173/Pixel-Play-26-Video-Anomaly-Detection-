# =========================================
# ResNet18 Feature-based Anomaly Detection
# Pixel Play / Avenue Dataset
# =========================================

import os, cv2
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import pandas as pd
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

TRAIN_ROOT = r"D:\pytorch\Avenue_Corrupted-20251221T112159Z-3-001\Avenue_Corrupted\Dataset\training_videos"
TEST_ROOT  = r"D:\pytorch\Avenue_Corrupted-20251221T112159Z-3-001\Avenue_Corrupted\Dataset\testing_videos"

# -------------------------
# Load pretrained ResNet18
# -------------------------
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet.fc = nn.Identity()
resnet = resnet.to(DEVICE)
resnet.eval()

# -------------------------
# Image preprocessing
# -------------------------
def preprocess(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.tensor(img / 255.).permute(2,0,1).float()
    return img.unsqueeze(0).to(DEVICE)

# -------------------------
# 1. Extract NORMAL features
# -------------------------
features = []

with torch.no_grad():
    for vid in tqdm(sorted(os.listdir(TRAIN_ROOT))):
        vpath = os.path.join(TRAIN_ROOT, vid)
        for f in sorted(os.listdir(vpath)):
            img = preprocess(os.path.join(vpath, f))
            feat = resnet(img).cpu().numpy().squeeze()
            features.append(feat)

features = np.array(features)
mean = features.mean(axis=0)
cov = np.cov(features, rowvar=False)
cov += 1e-6 * np.eye(cov.shape[0])  # numerical stability
inv_cov = np.linalg.inv(cov)

# -------------------------
# 2. Inference (Mahalanobis)
# -------------------------
results = []

def mahalanobis(x):
    diff = x - mean
    return np.sqrt(diff @ inv_cov @ diff.T)

with torch.no_grad():
    for vid in tqdm(sorted(os.listdir(TEST_ROOT))):
        vpath = os.path.join(TEST_ROOT, vid)
        for f in sorted(os.listdir(vpath)):
            frame_num = int(f.split("_")[-1].split(".")[0])
            img = preprocess(os.path.join(vpath, f))
            feat = resnet(img).cpu().numpy().squeeze()
            score = mahalanobis(feat)
            results.append([f"{int(vid)}_{frame_num}", score])

# -------------------------
# 3. Temporal smoothing
# -------------------------
df = pd.DataFrame(results, columns=["Id", "Predicted"])
df[["vid","frame"]] = df["Id"].str.split("_", expand=True).astype(int)
df = df.sort_values(["vid","frame"])

df["Predicted"] = (
    df.groupby("vid")["Predicted"]
      .rolling(window=7, center=True, min_periods=1)
      .mean()
      .reset_index(level=0, drop=True)
)

# Normalize
df["Predicted"] = (df["Predicted"] - df["Predicted"].min()) / \
                  (df["Predicted"].max() - df["Predicted"].min())

# -------------------------
# 4. Save CSV
# -------------------------
df[["Id","Predicted"]].to_csv(
    "submission_resnet.csv",
    index=False
)

print("Saved submission_resnet.csv")
print(df.head())
