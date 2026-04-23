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
| `capstone_credit_card_fraud_detection_final.ipynb` | Jupyter Notebook version - same pipeline with inline plots and markdown-sectioned cells |
| `data/creditcard.csv` | Kaggle dataset containing 284,807 credit card transactions |
| `README_final.md` | Project documentation |

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


---

## 3. Post EDA Process

### 3a. Preprocessing
- Scaled Amount and Time with RobustScaler (resistant to outliers)
- Engineered features:
  - `Amount_log` - log-transform to reduce skewness
  - `High_amount` - binary flag for transactions in the top 5% by amount
  - `best_interaction` - **data-driven interaction term**: systematically tested all pairs among the top 15 fraud-correlated features and selected the pair with the strongest correlation to fraud
- Stratified 80/20 train-test split

### 3b. Class Imbalance Handling
- Applied SMOTE-like oversampling (duplication + Gaussian noise) to bring fraud samples to ~50% of legitimate count in training data
- Before SMOTE: 227,451 legitimate vs 394 fraud
- After SMOTE:  227,451 legitimate vs 113,725 synthetic fraud

### 4. Model Training with Grid Search & Cross-Validation
Each model was tuned via **GridSearchCV** with **5-fold stratified cross-validation**, optimizing for Recall:

| Model | Hyperparameters Searched |
|-------|--------------------------|
| Logistic Regression | C: [0.01, 0.1, 1, 10]; class_weight: [balanced, None] |
| Random Forest | n_estimators: [100]; max_depth: [10, None]; class_weight: balanced_subsample |
| Gradient Boosting | n_estimators: [100]; max_depth: [3]; learning_rate: [0.1] |
| Neural Network (MLP) | hidden_layers: [(64, 32)]; learning_rate: [0.001, 0.01] |

### 5. Evaluation Metrics
- **Primary:** Recall (fraction of fraud caught) and AUPRC (robust to class imbalance)
- **Secondary:** Precision, F1-Score, ROC-AUC, Accuracy
- **Rationale:** Accuracy is misleading at 99.83% class imbalance - a model predicting all transactions as legitimate scores 99.83% accuracy but catches zero fraud. Recall and AUPRC directly measure what matters: detecting fraud.

### 6. Threshold Optimization
- The default 0.5 cutoff isn't always optimal. We sweep thresholds from 0.1 to 0.9 to find the cutoff that best balances catching fraud (recall) against false alarms (precision)
- For fraud detection, a lower threshold is usually better since missing fraud costs far more than investigating a false alarm

### 7. Cost-Benefit Analysis
- Estimated financial impact assuming each missed fraud costs the average fraud amount (~$122) and each false positive costs $15 in investigation overhead

---

## Key Findings

**1. The Biggest Challenge: Too Few Fraud Cases**
Only 0.17% of transactions are fraud. Without balancing the data first, models take the lazy shortcut - predicting every transaction as legitimate and catching zero fraud. SMOTE oversampling solved this by creating synthetic fraud examples.

**2. Smarter Models Beat Simpler Ones**
Random Forest and Gradient Boosting caught significantly more fraud than Logistic Regression. Fraud patterns involve complex combinations of features that a straight-line model can't capture. The Neural Network performed competitively but needed more tuning.

**3. The Right Features Matched Expectations**
V17, V14, V12, and V10 emerged as the strongest fraud indicators - consistent with published research on this dataset. Engineered features (Amount_log, the best interaction pair) added a small but real boost.

**4. Adjusting the Decision Cutoff Helped**
Moving from the default 0.5 threshold to an optimized value caught more fraud without a proportional increase in false alarms.

**5. The Business Case is Strong**
Deploying the best model saves far more in prevented fraud than it costs in investigating false alarms.

---

## What This Means for Non-Technical Teams

- The model acts as a **smart first filter** - it scans every transaction and flags ones that look suspicious based on patterns learned from history.
- A flag doesn't mean "block the card." It means "take a second look" - typically a text message to the customer asking "Did you just make this purchase?"
- The model is tuned to **err on the side of caution**. We'd rather inconvenience a customer with a quick verification text than let a real fraud slip through, because missed fraud costs the bank hundreds of dollars while a verification text costs almost nothing.

---

## Next Steps & Recommendations

1. **Try more models** - Test additional classification algorithms to see if any perform better on our data.
2. **Tune hyperparameters further** - Expand the hyperparameter search to explore more combinations.
3. **Try different resampling methods** - Experiment with alternatives to SMOTE to see if other balancing techniques catch more fraud.
4. **Combine our existing models** - Build a meta-ensemble that blends predictions from all four models (Logistic Regression, Random Forest, Gradient Boosting, Neural Network) for potentially stronger results.
5. **Add more features** - Explore additional feature engineering and feature selection techniques.
6. **Add more evaluation metrics** - Include extra metrics that are well-suited for imbalanced datasets.

---

## Libraries Used

- **Data:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Modeling:** scikit-learn (LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, MLPClassifier, GridSearchCV, StratifiedKFold, make_scorer)