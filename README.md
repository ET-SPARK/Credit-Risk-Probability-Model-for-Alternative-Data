# Credit Risk Probability Model for Alternative Data

## Task-1 Credit Scoring Business Understanding

### How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Accord emphasizes robust risk management frameworks, requiring financial institutions to accurately measure and manage credit risk. This necessitates models that are not only predictive but also highly interpretable and well-documented. Interpretability is crucial for regulators to understand the model's logic, validate its assumptions, and ensure it aligns with regulatory guidelines. Well-documented models provide transparency, facilitate independent review, and ensure reproducibility, all of which are vital for compliance and auditability under Basel II. Without interpretability and thorough documentation, models would be considered black boxes, making it difficult to justify capital allocation and risk provisions to regulatory bodies.

### Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

In many alternative data scenarios, a direct "default" label (e.g., a formal loan default) might be unavailable. Therefore, creating a proxy variable (e.g., severe delinquency, bankruptcy filings, or other adverse financial events) becomes necessary to train a credit risk model. This proxy acts as a substitute for the true default event.

However, relying on a proxy introduces significant business risks:

- **Misclassification Risk:** The proxy might not perfectly capture the true definition of default, leading to misclassification of borrowers. This could result in approving high-risk applicants or rejecting creditworthy ones.
- **Model Drift:** The relationship between the proxy and actual default could change over time, leading to model performance degradation.
- **Regulatory Scrutiny:** Regulators may question the validity of the proxy and its alignment with actual credit risk, potentially leading to non-compliance issues.
- **Financial Losses:** Inaccurate predictions based on a flawed proxy can lead to increased loan losses, reduced profitability, and reputational damage.

### What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

In a regulated financial context, the choice between simple, interpretable models and complex, high-performance models involves critical trade-offs:

**Simple, Interpretable Models (e.g., Logistic Regression with WoE):**

- **Pros:**
  - **Interpretability:** Easy to understand how each variable contributes to the prediction, crucial for regulatory approval and explaining decisions to customers.
  - **Transparency:** Clear logic allows for straightforward validation and auditing.
  - **Stability:** Less prone to overfitting and often more stable over time.
  - **Regulatory Acceptance:** Often preferred by regulators due to their transparency and ease of validation.
- **Cons:**
  - **Lower Performance:** May not capture complex non-linear relationships in the data, potentially leading to lower predictive accuracy compared to complex models.
  - **Feature Engineering Intensive:** Often requires extensive feature engineering (like WoE transformation) to achieve good performance.

**Complex, High-Performance Models (e.g., Gradient Boosting):**

- **Pros:**
  - **Higher Performance:** Can capture intricate patterns and non-linear relationships, leading to superior predictive accuracy.
  - **Less Feature Engineering:** Often require less manual feature engineering.
- **Cons:**
  - **Lack of Interpretability (Black Box):** Difficult to understand the exact reasoning behind predictions, posing challenges for regulatory compliance and explaining decisions.
  - **Transparency Issues:** Harder to validate and audit, increasing regulatory scrutiny.
  - **Overfitting Risk:** More prone to overfitting, especially with limited data, leading to poor generalization on unseen data.
  - **Model Risk:** Higher risk of undetected errors or biases due to their complexity.

In a regulated environment, the emphasis on interpretability, transparency, and regulatory acceptance often favors simpler models, even if it means sacrificing some predictive power. However, with advancements in explainable AI (XAI) techniques, the gap in interpretability for complex models is narrowing, potentially allowing for their increased adoption in the future, provided robust validation and explanation frameworks are in place.

## Project Structure

```
├── data
│   ├── processed
│   └── raw
├── notebooks
│   └── 1.0-eda.ipynb
├── src
│   ├── __init__.py
│   ├── api
│   │   ├── main.py
│   │   └── pydantic_models.py
│   ├── data_processing.py
│   ├── predict.py
│   └── train.py
├── tests
│   └── test_data_processing.py
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── LICENSE
├── README.md
└── requirements.txt
```

## Task 2 - Exploratory Data Analysis (EDA)

Based on the initial exploratory data analysis (EDA) performed in `notebooks/1.0-eda.ipynb`, here are the key insights:

1. **No Missing Values:** The dataset is clean with no apparent missing values, which simplifies the data preprocessing step.
2. **Negative Transaction Amounts:** The 'Amount' column contains negative values. These likely represent refunds or transaction reversals and require further investigation to understand their impact on fraud detection.
3. **Categorical Feature Encoding:** Several columns such as `ProductCategory`, `ChannelId`, `CurrencyCode`, and `CountryCode` are categorical. These features will need to be appropriately encoded (e.g., one-hot encoding) before being used in machine learning models.
4. **Fraud Result Distribution:** The target variable, `FraudResult`, needs to be carefully examined for class imbalance. An imbalanced distribution (where fraudulent transactions are rare) is common in fraud detection and will necessitate specific handling techniques (e.g., oversampling, undersampling, or specialized evaluation metrics) during model training.

## Task-3 Feature Engineering

The `src/data_processing.py` script is responsible for feature engineering. It takes raw data, performs various transformations, and saves the processed data and a scikit-learn pipeline.

### How to run the script

1.  Place your raw data in the `data/raw` directory. The data should be in a CSV file named `credit_risk_data.csv`.
2.  Run the script from the root of the project:

```bash
python src/data_processing.py
```

This will generate the processed data in `data/processed/processed_credit_risk_data.csv` and the pipeline in `src/pipeline.joblib`.

### Transformations

The script performs the following transformations:

- **Aggregate Features:** Creates features like total transaction amount, average transaction amount, etc.
- **Date Features:** Extracts features like hour, day, month, and year from the transaction date.
- **Categorical Encoding:** Uses one-hot encoding for categorical features.
- **Missing Value Imputation:** Fills missing values using the median for numerical features and the most frequent value for categorical features.
- **Normalization/Standardization:** Standardizes numerical features to have a mean of 0 and a standard deviation of 1.

## Task 4 - Proxy Target Variable Engineering

Since a direct "credit risk" column is not available, a proxy target variable named `is_high_risk` is engineered. This involves identifying a group of "disengaged" customers who are labeled as high-risk proxies, representing those with a high likelihood of default.

### Implementation in `src/data_processing.py`

The `src/data_processing.py` script has been updated to include the following steps for proxy target variable engineering:

1.  **Calculate RFM Metrics:** For each `customer_id`, Recency, Frequency, and Monetary (RFM) values are calculated from the transaction history. A snapshot date (one day after the latest transaction) is used for consistent Recency calculation.
2.  **Cluster Customers:** The K-Means clustering algorithm is applied to segment customers into 3 distinct groups based on their RFM profiles. RFM features are scaled using `StandardScaler` before clustering to ensure meaningful results. A `random_state` is set for reproducibility.
3.  **Define and Assign "High-Risk" Label:** The resulting clusters are analyzed to determine which one represents the least engaged and therefore highest-risk customer segment (typically characterized by high Recency, low Frequency, and low Monetary value). A new binary target column named `is_high_risk` is created, with a value of 1 for customers in this high-risk cluster and 0 for all others.
4.  **Integrate the Target Variable:** The `is_high_risk` column is merged back into the main processed dataset, making it available for subsequent model training.

## Task-5 Model Training

The `src/train.py` script is responsible for training the model. It loads the processed data and trains a logistic regression model.

### How to run the script

1.  Make sure you have run the data processing script first.
2.  Run the script from the root of the project:

```bash
python src/train.py
```

This will save the trained model to `src/model.joblib`.

## Prediction

The `src/predict.py` script is responsible for making predictions on new data. It loads the trained model and the pipeline and makes predictions on new data.

### How to run the script

1.  Place your new data in the `data/raw` directory. The data should be in a CSV file named `new_credit_risk_data.csv`.
2.  Run the script from the root of the project:

```bash
python src/predict.py
```

This will print the predictions to the console.
