import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------------------------------------------------------
# Data Loading & Initial Exploration
# ------------------------------------------------------------------

def load_data(file_path):
    """Loads data from a CSV file using Pandas. Handles potential errors."""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File at {file_path} is empty.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def explore_data(df):
    """Provides a comprehensive overview of the dataset."""
    print("\n--- Data Overview ---")
    print(f"Shape: {df.shape}")
    print("\n--- First 5 Rows ---")
    print(df.head())
    print("\n--- Data Types ---")
    print(df.dtypes)
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    print("\n--- Descriptive Statistics ---")
    print(df.describe())

# ------------------------------------------------------------------
# Data Preprocessing
# ------------------------------------------------------------------

def preprocess_data(df):
    """Handles missing values, converts data types, and performs feature scaling."""
    # Handle missing values (example: fill with mean for numerical columns)
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0]) #Mode for categorical columns
    # Convert data types (example: string to numeric)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce') # Handle potential conversion errors
        except ValueError:
            pass
    # Feature scaling (example: StandardScaler)
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=np.number).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df

# ------------------------------------------------------------------
# Model Training & Evaluation
# ------------------------------------------------------------------

def train_model(df, X, y, model_type='LogisticRegression'):
    """Trains and evaluates a machine learning model."""
    print(f"\n--- Training {model_type} ---")
    if model_type == 'LogisticRegression':
        model = LogisticRegression(solver='liblinear', random_state=42) # Add random_state for reproducibility
    else:
        print("Unsupported model type.")
        return None

    model.fit(X, y)
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y, y_pred))

    return model

# ------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------

def visualize_data(df):
    """Generates informative visualizations of the data."""
    # Histogram of numerical features
    for col in df.select_dtypes(include=np.number).columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Histogram of {col}')
        plt.show()

    # Scatter plots for numerical features
    for i in range(len(df.select_dtypes(include=np.number).columns)):
        for j in range(i + 1, len(df.select_dtypes(include=np.number).columns)):
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=df[df.columns[i]], y=df[df.columns[j]])
            plt.title(f'Scatter Plot of {df.columns[i]} vs {df.columns[j]}')
            plt.show()

    # Boxplots for categorical features
    for col in df.select_dtypes(include=['object']).columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.show()


# ------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------

def main():
    """Main function to orchestrate the data analysis process."""

    # 1. Load Data
    file_path = "your_data.csv"  # Replace with your CSV file path
    df = load_data(file_path)
    if df is None:
        return

    # 2. Explore Data
    explore_data(df)

    # 3. Preprocess Data
    df = preprocess_data(df)

    # 4. Split Data into Training and Testing Sets
    X = df.drop('target_column', axis=1)  # Replace 'target_column' with your target column
    y = df['target_column']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Train Model
    model = train_model(df, X_train, y_train, model_type='LogisticRegression')

    if model is not None:
        # 6. Evaluate Model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Final Accuracy: {accuracy:.2f}")

        print("\n--- Model Summary ---")
        print(model.summary())

    # 7. Visualize Data
    visualize_data(df)



if __name__ == "__main__":
    main()
