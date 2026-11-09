
# Student Score Prediction

A Machine Learning project focused on predicting students' examination scores using academic and behavioral factors.

## 1. Project Overview

The objective of this project is to develop predictive models capable of estimating students' examination scores based on different factors such as study hours, attendance, previous academic performance, sleep hours and tutoring sessions.

The project explores the relationship between these factors and academic performance through data analysis, visualization and regression techniques.

## 2. Dataset

The project uses the Student Performance Factors dataset.

**Dataset information:**
- Observations: 6,607 students
- Features: 19 explanatory variables
- Target variable: `Exam_Score`
- Numerical variables: 7
- Categorical variables: 13

The dataset contains academic, behavioral and personal information about students.

## 3. Technologies Used

- Python 3.11
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- VS Code

## 4. Data Exploration

Initial exploration was performed using:

- `df.shape` to inspect dataset dimensions.
- `df.info()` to examine column types and missing values.
- `df.describe()` to analyze numerical distributions.
- `df.isnull().sum()` to identify missing values.

The dataset initially contained 6,607 observations and 20 columns.

## 5. Data Cleaning

A total of 235 missing values were identified across three categorical variables.

| Feature | Missing Values |
|---|---:|
| Teacher_Quality | 78 |
| Parental_Education_Level | 90 |
| Distance_from_Home | 67 |
| **Total** | **235** |

Missing values were replaced using the mode of each respective column.

After imputation, no missing values remained in the dataset.

Note: Mode imputation does not recover the original missing information and may influence category distributions.

## 6. Outlier Detection

Potential outliers in `Exam_Score` were investigated using the Interquartile Range (IQR) method.

| Statistic | Value |
|---|---:|
| Q1 | 65 |
| Q3 | 69 |
| IQR | 4 |
| Lower Bound | 59 |
| Upper Bound | 75 |

A total of 104 potential outliers were identified, representing approximately 1.57% of the dataset.

These observations were retained rather than automatically removed because unusual values are not necessarily incorrect.

## 7. Exploratory Data Analysis (EDA)

The following techniques were used:

- Statistical analysis
- Missing value analysis
- Boxplot visualization
- Histogram visualization
- Scatter plot analysis
- Pearson correlation

The correlation coefficient between `Hours_Studied` and `Exam_Score` was:

**0.4455**

This indicates a moderate positive linear relationship between study hours and examination scores.

Correlation does not necessarily imply causation.

## 8. Machine Learning Models

Three regression approaches were evaluated.

### 8.1 Simple Linear Regression

The first model used only `Hours_Studied` to predict `Exam_Score`.

Its purpose was to establish a baseline predictive model.

### 8.2 Polynomial Regression

A second-degree polynomial transformation was applied to `Hours_Studied` to investigate whether a nonlinear relationship could improve predictive performance.

### 8.3 Multiple Linear Regression

The third model used six numerical explanatory variables:

- Hours_Studied
- Attendance
- Previous_Scores
- Sleep_Hours
- Tutoring_Sessions
- Physical_Activity

This model investigated whether combining multiple factors could improve predictions.

## 9. Model Training and Evaluation

The dataset was divided into:

- 80% training data
- 20% testing data

The same train-test split was used to compare all three models.

Four evaluation metrics were used:

- MAE: Mean Absolute Error
- MSE: Mean Squared Error
- RMSE: Root Mean Squared Error
- R²: Coefficient of Determination

### Model Performance Comparison

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Simple Linear Regression | 2.4476 | 10.8559 | 3.2948 | 0.2320 |
| Polynomial Regression | 2.4448 | 10.8450 | 3.2932 | 0.2328 |
| Multiple Linear Regression | **1.2659** | **5.0655** | **2.2507** | **0.6416** |

### Best Performing Model

Multiple Linear Regression achieved the best performance among the three evaluated models.

Its R² score of 0.6416 indicates that it explains approximately 64.16% of the variation in examination scores in the test dataset.

Its MAE of 1.2659 indicates an average absolute prediction error of approximately 1.27 examination points.

The results suggest that using multiple academic and behavioral factors provides substantially more predictive information than using study hours alone.

## 10. Multiple Regression Coefficients

The multiple regression model produced the following coefficients:

| Feature | Coefficient |
|---|---:|
| Tutoring_Sessions | +0.5099 |
| Hours_Studied | +0.2893 |
| Attendance | +0.1987 |
| Physical_Activity | +0.1511 |
| Previous_Scores | +0.0482 |
| Sleep_Hours | -0.0335 |

These coefficients describe the estimated relationship between each feature and the predicted examination score, while holding the other selected features constant.

They should not be interpreted as proof of causation or as a definitive ranking of feature importance.

## 11. Visualizations

The project generates the following visualizations:

- `exam_score_boxplot.png` — examination score distribution and potential outliers.
- `simple_linear_regression.png` — actual scores and the simple regression line.
- `multiple_regression_predictions.png` — actual versus predicted examination scores.
- `model_comparison.png` — comparison of R² scores across the three models.

## 12. Project Structure

```text
student-score-prediction/
│
├── data/
│   └── StudentPerformanceFactors (1).csv
│
├── main.py
├── README.md
├── .gitignore
│
├── exam_score_boxplot.png
├── simple_linear_regression.png
├── multiple_regression_predictions.png
└── model_comparison.png
```

The dataset is excluded from version control through `.gitignore`.

## 13. Limitations and Future Improvements

Several improvements could be explored in future iterations:

- Encoding categorical variables to incorporate additional information.
- Applying feature scaling and standardized coefficient analysis.
- Using cross-validation to assess model stability.
- Investigating alternative regression algorithms.
- Performing further analysis of potential outliers.
- Comparing model performance across different student groups.

## 14. Conclusion

This project demonstrates the application of Python and Machine Learning techniques to student examination score prediction.

The initial simple linear regression model showed that study hours alone have limited predictive power.

Polynomial regression produced only a marginal improvement.

Multiple linear regression achieved substantially better performance, reaching an R² score of 0.6416 and an MAE of approximately 1.27 points.

The findings suggest that combining several academic and behavioral factors provides a more informative basis for predicting examination scores.

This project provided practical experience in data cleaning, exploratory data analysis, visualization, regression modeling and model evaluation.

---

**Author:** Ilobakima Robert

**GitHub:** [loisilobakima](https://github.com/loisilobakima)
