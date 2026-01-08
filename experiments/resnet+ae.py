import pandas as pd

# Load submissions
df_ae = pd.read_csv(r"D:\pytorch\submission_smoothed.csv")
df_res = pd.read_csv(r"D:\pytorch\submission_resnet.csv")

# Safety check
assert len(df_ae) == len(df_res)
assert (df_ae["Id"] == df_res["Id"]).all()

# Weighted fusion
# AE slightly stronger for Avenue
ALPHA = 0.65   # AE weight
BETA  = 0.35   # ResNet weight

df_final = pd.DataFrame()
df_final["Id"] = df_ae["Id"]
df_final["Predicted"] = (
    ALPHA * df_ae["Predicted"].values +
    BETA  * df_res["Predicted"].values
)

# Normalize
df_final["Predicted"] = (
    df_final["Predicted"] - df_final["Predicted"].min()
) / (
    df_final["Predicted"].max() - df_final["Predicted"].min()
)

# Save final submission
df_final.to_csv(
    "submission_FINAL.csv",
    index=False
)

print("Saved submission_FINAL.csv")
print(df_final.head())