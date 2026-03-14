import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Data Loading & Preparation
# --------------------------------------------------

def load_and_prepare_data(file_path):
    """Loads the data, handles missing values, and prepares it for the models."""
    df = pd.read_csv(file_path)

    # Handle missing values (replace with mean for numerical, mode for categorical)
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])  # Mode for categorical

    # Convert categorical features to numerical (one-hot encoding)
    df = pd.get_dummies(df, columns=df.columns)

    # Feature scaling (important for some models)
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=np.number).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df



# --------------------------------------------------
# 2. Model Training
# --------------------------------------------------

def train_fraud_detection_models(df, test_size=0.2):
    """Trains Isolation Forest and Random Forest models."""

    # Split data into training and testing sets
    X = df.drop('is_fraud', axis=1)  # Features
    y = df['is_fraud']  # Target variable (0 or 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train Isolation Forest
    if isolation_forest:
        isolation_forest.fit(X_train)
        y_pred_if = isolation_forest.predict(X_test)

    # Train Random Forest
    if random_forest:
        random_forest.fit(X_train, y_train)
        y_pred_rf = random_forest.predict(X_test)

    return isolation_forest, random_forest, y_pred_if, y_pred_rf



# --------------------------------------------------
# 3. Model Evaluation
# --------------------------------------------------

def evaluate_models(y_true, y_pred_if, y_pred_rf):
    """Evaluates the models using accuracy and classification report."""
    # Evaluate Isolation Forest
    accuracy_if = accuracy_score(y_true, y_pred_if)
    print(f"Isolation Forest Accuracy: {accuracy_if:.4f}")

    # Evaluate Random Forest
    accuracy_rf = accuracy_score(y_true, y_pred_rf)
    print(f"Random Forest Accuracy: {accuracy_rf:.4f}")

    # Print classification report for Random Forest
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_true, y_pred_rf))



# --------------------------------------------------
# 4. Main Execution
# --------------------------------------------------

if __name__ == '__main__':
    # Load Data
    file_path = 'fraud_data.csv'  # Replace with your data file
    df = load_and_prepare_data(file_path)


    # Initialize Models
    isolation_forest = IsolationForest(n_estimators=100, random_state=42)
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train Models
    isolation_forest, random_forest, y_pred_if, y_pred_rf = train_fraud_detection_models(df)

    # Evaluate Models
    evaluate_models(df['is_fraud'], y_pred_if, y_pred_rf)
