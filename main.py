import pandas as pd

print("Starting Student Score Prediction project...")

# Load dataset
df = pd.read_csv("data/StudentPerformanceFactors (1).csv")

# Explore the dataset
print("\nDataset dimensions:")
print(df.shape)

print("\nDataset information:")
print(df.info())

print("\nStatistical summary:")
print(df.describe())