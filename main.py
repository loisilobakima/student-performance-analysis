import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
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

plt.figure()

plt.hist(df["Exam_Score"],bins=10)
plt.title(" histogram exam score")
plt.xlabel("exam score")
plt.ylabel("nombre d'etudiant")
plt.show()

plt.scatter(df["Hours_Studied"], df["Exam_Score"])
plt.title("nuage de point entre heure etude et scor examen")
plt.xlabel("hours study")
plt.ylabel("scor exam")
plt.show()

coefficient_correlation=df["Hours_Studied"].corr(df["Exam_Score"])
print("\n coefficient de correlation entre heure etude et score examen",coefficient_correlation)

x = df[[
    "Hours_Studied",
]] 
# ==========================================
# MACHINE LEARNING - MODEL TRAINING
# ==========================================

# Target variable
y = df["Exam_Score"]

# Features for multiple linear regression
x_multiple = df[[
    "Hours_Studied",
    "Attendance",
    "Previous_Scores",
    "Sleep_Hours",
    "Tutoring_Sessions",
    "Physical_Activity"
]]

# Split the dataset
# All three models will use the same train/test observations
x_train_multiple, x_test_multiple, y_train, y_test = train_test_split(
    x_multiple,
    y,
    test_size=0.2,
    random_state=42
)

# Extract the single feature for simple and polynomial regression
x_train_simple = x_train_multiple[["Hours_Studied"]]
x_test_simple = x_test_multiple[["Hours_Studied"]]


# ==========================================
# MODEL 1 - SIMPLE LINEAR REGRESSION
# ==========================================

simple_model = LinearRegression()

simple_model.fit(x_train_simple, y_train)

y_pred_simple = simple_model.predict(x_test_simple)


# ==========================================
# MODEL 2 - POLYNOMIAL REGRESSION
# ==========================================

poly = PolynomialFeatures(degree=2, include_bias=False)

x_train_poly = poly.fit_transform(x_train_simple)
x_test_poly = poly.transform(x_test_simple)

poly_model = LinearRegression()

poly_model.fit(x_train_poly, y_train)

y_pred_poly = poly_model.predict(x_test_poly)


# ==========================================
# MODEL 3 - MULTIPLE LINEAR REGRESSION
# ==========================================

multiple_model = LinearRegression()

multiple_model.fit(x_train_multiple, y_train)

y_pred_multiple = multiple_model.predict(x_test_multiple)


# ==========================================
# MODEL EVALUATION
# ==========================================

models_predictions = {
    "Simple Linear Regression": y_pred_simple,
    "Polynomial Regression": y_pred_poly,
    "Multiple Linear Regression": y_pred_multiple
}

results = []

for model_name, predictions in models_predictions.items():

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

results_df = pd.DataFrame(results)

print("\n==========================================")
print("FINAL MODEL COMPARISON")
print("==========================================")

print(results_df.round(4).to_string(index=False))


# ==========================================
# MODEL COEFFICIENTS
# ==========================================

coefficients = pd.DataFrame({
    "Feature": x_train_multiple.columns,
    "Coefficient": multiple_model.coef_
})

coefficients["Absolute_Coefficient"] = coefficients["Coefficient"].abs()

print("\n==========================================")
print("MULTIPLE REGRESSION COEFFICIENTS")
print("==========================================")

print(
    coefficients
    .sort_values(by="Absolute_Coefficient", ascending=False)
    .round(4)
    .to_string(index=False)
)

print("\nIntercept:", round(multiple_model.intercept_, 4))


# ==========================================
# VISUALIZATION 1 - SIMPLE REGRESSION
# ==========================================

x_line = pd.DataFrame({
    "Hours_Studied": np.linspace(
        x_train_simple["Hours_Studied"].min(),
        x_train_simple["Hours_Studied"].max(),
        100
    )
})

y_line = simple_model.predict(x_line)

plt.figure(figsize=(10, 6))

plt.scatter(
    x_test_simple["Hours_Studied"],
    y_test,
    alpha=0.5,
    label="Actual Scores"
)

plt.plot(
    x_line["Hours_Studied"],
    y_line,
    label="Regression Line"
)

plt.title("Simple Linear Regression")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.legend()
plt.tight_layout()
plt.savefig("simple_linear_regression.png")
plt.show()


# ==========================================
# VISUALIZATION 2 - MULTIPLE REGRESSION
# ACTUAL VS PREDICTED SCORES
# ==========================================

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    y_pred_multiple,
    alpha=0.5
)

min_score = min(y_test.min(), y_pred_multiple.min())
max_score = max(y_test.max(), y_pred_multiple.max())

plt.plot(
    [min_score, max_score],
    [min_score, max_score],
    linestyle="--",
    label="Perfect Predictions"
)

plt.title("Multiple Regression - Actual vs Predicted")
plt.xlabel("Actual Exam Scores")
plt.ylabel("Predicted Exam Scores")
plt.legend()
plt.tight_layout()
plt.savefig("multiple_regression_predictions.png")
plt.show()


# ==========================================
# VISUALIZATION 3 - MODEL COMPARISON
# ==========================================

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["R2"]
)

plt.title("R² Comparison Between Models")
plt.ylabel("R² Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()


print("\nProject model training and evaluation completed.")
