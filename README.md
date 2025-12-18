# Telco Customer Churn Prediction using Random Forest

## **Project Overview**

Customer churn is a critical business problem in subscription-based industries. This project focuses on predicting customer churn using the **Telco Customer Churn dataset** and applying a **Random Forest classification model** with **target encoding**, **out-of-bag (OOB) validation**, and **decision threshold optimization** to align predictions with real-world business objectives.

The goal is not just high accuracy, but **early identification of customers at risk of churn** to enable proactive retention strategies.

---

## **Dataset**

- **Source**: Kaggle – Telco Customer Churn Dataset
- **Records**: 7,043 customers
- **Features**: 20 (after removing non-informative identifiers)
- **Target Variable**: `Churn` (Binary: 1 = Yes, 0 = No)
- The dataset contains a mix of demographic, service-related, billing, and contract information.

---

## **Data Cleaning & Preprocessing**

- Removed `customerID` as it provides no predictive value.
- Converted `TotalCharges` to numeric and corrected zero-tenure cases.
- Encoded the target variable (`Churn`) into binary format.
- Identified and handled feature redundancy (e.g., `tenure` vs `TotalCharges`).
- Applied target encoding for categorical variables to retain churn-risk information without high dimensionality.

---

## **Exploratory Data Analysis (EDA)**

Key findings from EDA:
- **Tenure** is the strongest predictor: churn is highest in the first year and declines steadily over time.
- **Month-to-month contracts** show significantly higher churn than long-term contracts.
- Customers using fiber **optic internet, electronic check payments**, and **paperless billing** churn more frequently.
- Value-added services such as **OnlineSecurity**, **TechSupport**, and **DeviceProtection** are associated with lower churn.
- Demographic features (e.g., gender) have minimal predictive power.
- Linear correlations with churn are weak, confirming the need for non-linear models.

---

## **Model Building**

- Model: Random Forest Classifier
- Encoding: Target Encoding for categorical features
- Class Imbalance Handling: `class_weight='balanced'`
- Validation: Out-of-Bag (OOB) score
- Hyperparameter Tuning:
  - `min_samples_leaf = 25`
  - `max_depth = 15`


### **Model Performance**

- OOB Score: ~0.76
- ROC-AUC: ~0.84
- Accuracy: ~0.78

These results indicate strong ranking performance and good generalization.

---

## **Threshold Optimization**

Instead of using the default 0.5 threshold, multiple thresholds were evaluated.
- Selected Threshold: `0.35`
- Reason: Maximizes churn recall (~67%) while maintaining acceptable precision.
- This aligns with business goals where missing a churner is more costly than flagging a loyal customer.
- The threshold was intentionally kept outside the model to allow business flexibility.

---

## **Evaluation & Interpretability**

- Confusion matrix and classification metrics were analyzed at the optimized threshold.
- Feature importance analysis showed:
  - Tenure, Contract type, InternetService, and OnlineSecurity among top predictors.
- Results were validated using both OOB and hold-out test performance.

---

## **Key Takeaways**

- Churn is driven more by behavioral and contractual factors than demographics.
- Tree-based models are well-suited for churn prediction due to non-linear interactions.
- Threshold tuning is essential for aligning ML models with business impact.
- Separating probability estimation (model) from decision-making (threshold) is a best practice.

---

## **Future Improvements**

- Cost-sensitive evaluation based on retention campaign ROI
- Probability calibration for risk scoring
- Model comparison with Gradient Boosting or XGBoost
- Deployment as a real-time churn risk monitoring system

---

## **File Structure**

```
├── Telco_Customer_Churn.csv
├── Telco_Customer_Churn.ipynb
├── README.md
```

----

## **Running the Project**

1. Clone this repository:
```bash
git clone https://github.com/PranayDomal/Random_Forest_Classification.git
```
2. Navigate to the project directory.
3. Open and run the Jupyter Notebook:
```bash
jupyter notebook Telco_Customer_Churn.ipynb
```

---

## **Author**

https://www.linkedin.com/in/pranay-domal-a641bb368/
