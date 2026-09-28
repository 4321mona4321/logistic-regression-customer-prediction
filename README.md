# logistic-regression-customer-prediction
# Binary Logistic Regression Classifier - Customer Purchase Prediction

## Project Overview
This repository contains a complete binary logistic regression pipeline built in Python to predict whether a customer will purchase a product (`PurchaseStatus`: 1 = Purchased, 0 = Did Not Purchase) based on demographic, behavioural, and historical session data.

The project follows the **CRISP methodology** and satisfies four essential model evaluation checkpoints:
1. **Binary Target Verification:** $Y \in \{0, 1\}$.
2. **Class Imbalance Audit & Adjustment:** Handled using stratified train-test splitting and `class_weight='balanced'`.
3. **Comprehensive Metric Evaluation:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC.
4. **Decision Threshold Sensitivity Analysis:** Tuning decision boundaries ($0.20$ to $0.80$) to evaluate Precision-Recall trade-offs.

---

## Dataset Overview
* **Dataset Name:** `customer_purchase_data.csv`
* **Total Instances:** 1,500 rows
* **Target Variable:** `PurchaseStatus`
  * `0`: Did Not Purchase (56.8%)
  * `1`: Purchased (43.2%)

### Predictor Features
| Feature | Type | Description |
|---|---|---|
| `Age` | Numerical | Customer age in years |
| `AnnualIncome` | Numerical | Annual income in USD |
| `NumberOfPurchases` | Numerical | Historical total purchases |
| `TimeSpentOnWebsite` | Numerical | Session duration in minutes |
| `DiscountsAvailed` | Numerical | Count of discounts used |
| `Gender` | Categorical | Customer gender |
| `ProductCategory` | Categorical | Product category code (0–3) |
| `LoyaltyProgram` | Categorical | Membership status (0 = No, 1 = Yes) |

---

## 🛠️ Pipeline Architecture
1. **Preprocessing (`ColumnTransformer`):**
   * Continuous variables scaled via `StandardScaler()`.
   * Categorical features encoded via `OneHotEncoder(drop='if_binary')`.
2. **Data Splitting:** 80/20 train-test split (`random_state=42`, `stratify=y`).
3. **Model Configuration:** `LogisticRegression(max_iter=1000, class_weight='balanced')`.

---

## 📈 Model Performance & Evaluation

### Default Threshold Results ($\tau = 0.50$)
* **Accuracy:** 83.67%
* **Precision:** 80.00%
* **Recall:** 83.08%
* **F1-Score:** 81.51%
* **ROC-AUC Score:** 0.8999

---

## 🎛️ Threshold Sensitivity Analysis
Evaluating probability cutoffs from $\tau = 0.20$ to $\tau = 0.80$:

| Threshold ($\tau$) | Precision | Recall | F1-Score | Accuracy |
|:---:|:---:|:---:|:---:|:---:|
| 0.20 | 0.5408 | 0.9846 | 0.6975 | 0.6133 |
| 0.30 | 0.6486 | 0.9231 | 0.7619 | 0.7300 |
| 0.40 | 0.7222 | 0.9000 | 0.8014 | 0.8000 |
| **0.50** | **0.8000** | **0.8308** | **0.8151** | **0.8367** |
| 0.60 | 0.8750 | 0.7538 | 0.8100 | 0.8467 |
| 0.70 | 0.9080 | 0.6077 | 0.7281 | 0.8033 |
| 0.80 | 0.9667 | 0.4462 | 0.6105 | 0.7538 |

> **Key Takeaway:** 
> * **Lowering the threshold ($\tau = 0.30/0.40$)** increases **Recall** up to **92.31%**, capturing nearly all potential buyers for wide marketing outreach.
> * **Raising the threshold ($\tau = 0.60$)** boosts **Precision** to **87.50%**, ensuring high confidence when issuing expensive promotions or targeted incentives.

---

## How to Run
```bash
# Clone repository
git clone 

# Install requirements
pip install pandas numpy scikit-learn matplotlib seaborn

# Run script
python logistic_regression.py
