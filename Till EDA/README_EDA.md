# Berkeley HAAS Professional Certificate in ML & AI: Professional Certificate in Machine Learning and Artificial Intelligence - Capstone Project

## Project Title
**Credit Card Application Fraud Detection using Machine Learning**

| | |
|---|---|
| **Domain** | FinTech / Financial Risk Management |
| **Date** | March 2026 |
| **Dataset** | [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |

---

## Project Structure

| File | Description |
|------|-------------|
| `capstone_credit_card_fraud_detection_EDA.ipynb` | Jupyter Notebook version - same pipeline with inline plots and markdown-sectioned cells |
| `data/creditcard.csv` | Kaggle dataset containing 284,807 credit card transactions |
| `README_EDA.md` | Project documentation |

---

## 1. Data Loading

The dataset is loaded from `data/creditcard.csv` using:

### Dataset Overview

| Property | Value |
|----------|-------|
| Total transactions | 284,807 |
| Legitimate transactions | 284,315 (99.83%) |
| Fraudulent transactions | 492 (0.17%) |
| Total features | 31 |
| Feature columns | Time, V1–V28 (PCA-transformed), Amount, Class |
| Target column | `Class` (0 = Legitimate, 1 = Fraud) |

The V1–V28 features are the result of **PCA (Principal Component Analysis)** applied to the original transaction features. The actual feature names were anonymized for confidentiality.

### Initial Data Inspection

- Verified dataset shape, data types, and basic statistics using `df.describe()`
- Confirmed **no missing values** across all columns
- Key statistics computed for Amount:
  - Legitimate transactions: avg ~$88
  - Fraudulent transactions: avg ~$122
  - Highest fraud transaction: $2,125

---

## 2. Exploratory Data Analysis (EDA)

### 2a. Class Distribution (Pie Chart)

Visualized the extreme class imbalance - 99.83% legitimate vs 0.17% fraud. This severe imbalance confirms the need for resampling techniques (SMOTE) before model training, as standard classifiers would be biased toward predicting all transactions as legitimate.

### 2b. Transaction Amount Distribution (Histogram)

Plotted the number of transactions vs transaction amount for each class separately:
- **Legitimate transactions** cluster heavily around lower amounts and taper off
- **Fraudulent transactions** show a wider spread with slightly higher average amounts
- Significant overlap exists between the two classes, meaning Amount alone is not sufficient for fraud detection

### 2c. Transaction Time Distribution (Histogram)

Analyzed when transactions occur over the 48-hour observation window:
- **Legitimate transactions** follow a cyclical pattern - peaks during daytime, dips at night - reflecting normal consumer behavior
- **Fraudulent transactions** are more randomly distributed across time, suggesting fraudsters do not follow typical business-hour patterns

### 2d. Correlation Heatmap

**Dynamically identified** the top correlated features were from the data:

**Top 15 Features Most Correlated with Fraud:**

| Rank | Feature | Correlation |
|------|---------|-------------|
| 1 | V17 | 0.3265 |
| 2 | V14 | 0.3025 |
| 3 | V12 | 0.2606 |
| 4 | V10 | 0.2169 |
| 5 | V16 | 0.1965 |
| 6 | V3 | 0.1930 |
| 7 | V7 | 0.1873 |
| 8 | V11 | 0.1549 |
| 9 | V4 | 0.1334 |
| 10 | V18 | 0.1115 |
| 11 | V1 | 0.1013 |
| 12 | V9 | 0.0977 |
| 13 | V5 | 0.0950 |
| 14 | V2 | 0.0913 |
| 15 | V6 | 0.0436 |


The correlation heatmap was generated using the top features along with Time, Amount and Class.

### 2e. Box Plots - Feature Distribution by Class

Created box plots for the top 18 features to visualize how feature values differ between legitimate and fraudulent transactions. Key observations:
- Features like **V17, V14, V12** show clear separation between the two classes - fraud transactions have distinctly different distributions
- **Amount** shows higher median and wider spread for fraud
- These visual differences confirm these features will be useful predictors for the ML models

---

## Key EDA Takeaways

1. **Severe class imbalance** (0.17% fraud) - must be addressed before model training
2. **Amount is a useful but insufficient feature** - fraud tends to involve higher amounts, but with significant overlap
3. **Time patterns differ** - fraud is randomly distributed while legitimate transactions follow day/night cycles
4. **V17, V14, V12, V10** are the strongest fraud indicators among the PCA features
5. **No missing data** - the dataset is clean and ready for preprocessing

---

## Next Steps (Covered in Full Notebook)

- Data Preprocessing (scaling, feature engineering)
- SMOTE oversampling to handle class imbalance
- Model Training (Logistic Regression, Random Forest, XGBoost, Neural Network)
- Model Evaluation & Comparison
- Threshold Optimization
- Cost-Benefit Analysis
