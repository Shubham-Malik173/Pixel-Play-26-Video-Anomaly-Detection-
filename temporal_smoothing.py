import pandas as pd

# Load existing submission
df = pd.read_csv(r"D:\pytorch\submission.csv")

# Split Id into video and frame
df[["vid", "frame"]] = df["Id"].str.split("_", expand=True).astype(int)
df = df.sort_values(by=["vid", "frame"])

# Temporal smoothing
df["Predicted"] = (
    df.groupby("vid")["Predicted"]
      .rolling(window=7, min_periods=1, center=True)
      .mean()
      .reset_index(level=0, drop=True)
)

# Re-normalize scores
df["Predicted"] = (
    df["Predicted"] - df["Predicted"].min()
) / (
    df["Predicted"].max() - df["Predicted"].min()
)

# Save new submission
df[["Id", "Predicted"]].to_csv(
    "submission_smoothed.csv",
    index=False
)

print(df.head())
