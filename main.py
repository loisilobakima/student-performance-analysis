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


print("\nvaleurs manquantes:")
print(df.isnull().sum())
print("\nmode de la colonne teacher quality")
print(df["Teacher_Quality"].mode())

print("\n nombre de valeur pour chaque categorie dans teacher quality")
print(df["Teacher_Quality"].value_counts())

df["Teacher_Quality"]=df["Teacher_Quality"].fillna(
    df["Teacher_Quality"].mode()[0]
)

print("\n nombre valeur manquante dans teacher quality")
print(df["Teacher_Quality"].isnull().sum())


print("\n nombre de valeur dans parental education level")
print(df["Parental_Education_Level"].value_counts())

print("\n nombre de valeur manquante dans parental education level")
print(df["Parental_Education_Level"].isnull().sum())

print("\n mode de la colonne parental education level")
print(df["Parental_Education_Level"].mode()[0])

df["Parental_Education_Level"] = df["Parental_Education_Level"].fillna(
    df["Parental_Education_Level"].mode()[0]
)


print("\n nombre de valeur manquante dans parental education level")
print(df["Parental_Education_Level"].isnull().sum())



print(df["Parental_Education_Level"].value_counts()
    )

print(df["Distance_from_Home"].value_counts())



print(df["Distance_from_Home"].mode()[0])

df["Distance_from_Home"]=df["Distance_from_Home"].fillna(
    df["Distance_from_Home"].mode()[0]
)

print("\n nombre de valeur manquante dans distance from home")
print(df["Distance_from_Home"].isnull().sum())


print("\n nnombre de donne manquante dans toute le data frame")
print(df.isnull().sum())