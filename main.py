import pandas as pd
import matplotlib.pyplot as plt

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


q1=df["Exam_Score"].quantile(0.25)
q3=df["Exam_Score"].quantile(0.75)

print("\n q1:",q1)
print("\n q3:" ,q3)

iqr=q3-q1

print("\n iqr:" , iqr)

limit_basse=q1-1.5*iqr

limit_haute=q3+1.5*iqr

print("\n limit_basse:" , limit_basse)
print("\n limit_haute:" , limit_haute)


outliers = df[
    (df["Exam_Score"] < limit_basse) |
    (df["Exam_Score"] > limit_haute)
]

print(outliers["Exam_Score"].value_counts())

percentage = (outliers.shape[0] / df.shape[0]) * 100

print("Percentage of outliers:", percentage, "%")


print("\n figure boxplot")

plt.figure()

plt.boxplot(df["Exam_Score"])
plt.savefig("exam_score_boxplot.png")
plt.show()