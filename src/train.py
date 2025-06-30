import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import os
from joblib import load

def train_and_log_model(model, X_train, y_train, X_test, y_test, model_name, params=None):
    """Trains a model, logs metrics and parameters to MLflow, and returns the trained model."""
    with mlflow.start_run(run_name=model_name) as run:
        if params:
            mlflow.log_params(params)
            model.set_params(**params)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        })
        mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
        
        print(f"--- {model_name} Metrics ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print("-" * 30)
        
        return model, {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1, "roc_auc": roc_auc}

if __name__ == '__main__':
    processed_data_path = 'data/processed/processed_credit_risk_data.csv'
    
    if not os.path.exists(processed_data_path):
        print("Processed data not found. Please run src/data_processing.py first.")
    else:
        df = pd.read_csv(processed_data_path)
        
        # Assuming 'is_high_risk' is the target variable
        X = df.drop(columns=['is_high_risk']) # Drop the original 'target' column as well
        y = df['is_high_risk']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # --- Logistic Regression ---
        lr_model = LogisticRegression(random_state=42, solver='liblinear')
        lr_params = {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2']
        }
        lr_grid = GridSearchCV(lr_model, lr_params, cv=3, scoring='roc_auc', n_jobs=-1)
        lr_grid.fit(X_train, y_train)
        best_lr_model, lr_metrics = train_and_log_model(lr_grid.best_estimator_, X_train, y_train, X_test, y_test, "LogisticRegression", lr_grid.best_params_)

        # --- Random Forest ---
        rf_model = RandomForestClassifier(random_state=42)
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_leaf': [1, 5]
        }
        rf_grid = GridSearchCV(rf_model, rf_params, cv=3, scoring='roc_auc', n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        best_rf_model, rf_metrics = train_and_log_model(rf_grid.best_estimator_, X_train, y_train, X_test, y_test, "RandomForestClassifier", rf_grid.best_params_)

        # --- Identify and Register Best Model ---
        if lr_metrics['roc_auc'] > rf_metrics['roc_auc']:
            best_model_name = "LogisticRegression"
            best_model = best_lr_model
            best_metrics = lr_metrics
        else:
            best_model_name = "RandomForestClassifier"
            best_model = best_rf_model
            best_metrics = rf_metrics
        
        print(f"Best model identified: {best_model_name} with ROC-AUC: {best_metrics['roc_auc']:.4f}")
        
        # Register the best model in MLflow Model Registry
        with mlflow.start_run(run_name="Best Model Registration"):
            mlflow.sklearn.log_model(best_model, "best_model", registered_model_name="CreditRiskModel")
            mlflow.log_metrics(best_metrics)
            print(f"Best model '{best_model_name}' registered as 'CreditRiskModel' in MLflow Model Registry.")
