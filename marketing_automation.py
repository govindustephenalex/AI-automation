import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1. Data Loading & Preprocessing
# --------------------------------------------------

def load_and_preprocess_data(file_path):
    """Loads data, handles missing values, and performs scaling."""
    df = pd.read_csv(file_path)

    # Handle missing values (example - replace with median for numerical)
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Data type conversions
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)  # Convert to string to handle mixed data
            
    # Feature scaling (standardization)
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=np.number).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df


# --------------------------------------------------
# 2. Customer Segmentation (K-Means)
# --------------------------------------------------

def customer_segmentation(df, n_clusters=3):
    """Performs K-Means clustering on customer data."""
    # Select features for clustering (customize based on your data)
    features = ['age', 'income', 'spending_score', 'website_visits'] #Example features
    X = df[features]
    
    # Scale features before clustering
    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto') #Ensure n_init is set
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    return df



# --------------------------------------------------
# 3. AI Recommendation System (Simple Neural Network)
# --------------------------------------------------

def build_recommendation_model(n_items=10):
    """Builds a simple neural network recommendation system."""
    model = Sequential()
    model.add(Dense(16, input_dim=df.shape[1], activation='relu')) #Dynamic input dimension
    model.add(Dense(8, activation='relu'))
    model.add(Dense(n_items, activation='softmax')) # Output layer for recommendations

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# --------------------------------------------------
# 4. Training & Evaluation
# --------------------------------------------------

def train_and_evaluate(model, X, y):
    """Trains the model and evaluates its performance."""
    model.fit(X, y, epochs=10, batch_size=8, validation_split=0.2)
    _, accuracy = model.evaluate(X, y, verbose=0)
    return model, accuracy



# --------------------------------------------------
# 5. Main Execution
# --------------------------------------------------

if __name__ == '__main__':
    # Load Data
    file_path = 'customer_data.csv'  # Replace with your data file
    df = load_and_preprocess_data(file_path)

    # Customer Segmentation
    df = customer_segmentation(df)
    print(df.head())

    # Prepare Data for Recommendation Model
    X = df.drop('cluster', axis=1)
    y = df['cluster'] #One-hot encode 'cluster' for categorical target variable
    y = pd.get_dummies(y, prefix='cluster')

    # Build and Train Recommendation Model
    n_items = 5 # Example number of items to recommend
    model = build_recommendation_model(n_items)
    model = train_and_evaluate(model, X, y)
    print(f"Recommendation Model Accuracy: {model[1]:.4f}")


    # Example Usage (Dummy Recommendation)
    # In a real application, you'd generate recommendations based on user features
    # Here, we just demonstrate the model's output.  You'd replace this with your logic
    # to predict a user's preferred cluster based on their data.
    # (Not a full recommendation system - just demonstration)

    # Example: predict cluster for a new user with features similar to those used for training
    # new_user_features = np.array([[50, 60000, 70, 5]]) # Example features
    # predicted_cluster = model.predict(new_user_features)[0]
    # print(f"Predicted Cluster for New User: {predicted_cluster}")
