import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load and preprocess the data
print("Loading and preprocessing data...")
df = pd.read_csv('framingham.csv')

# Handle missing values
imputer = SimpleImputer(strategy='median')
df_numeric = df.select_dtypes(include=[np.number])
df[df_numeric.columns] = imputer.fit_transform(df_numeric)

# Separate features and target
X = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']

# Save original feature names before encoding
original_features = list(X.columns)

# Convert categorical variables to numeric
X = pd.get_dummies(X, drop_first=True)

# Save the encoded feature names
encoded_features = list(X.columns)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame to maintain feature names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42, probability=True),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Dictionary to store results
results = {}

# Train and evaluate each model
print("\nTraining and evaluating models...")
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }
    
    print(f"{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(results[name]['classification_report'])

# Perform GridSearchCV for Random Forest
print("\nPerforming GridSearchCV for Random Forest...")
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), 
                      rf_params, 
                      cv=5, 
                      scoring='roc_auc',
                      n_jobs=-1)
rf_grid.fit(X_train_scaled, y_train)

print("\nBest Random Forest Parameters:", rf_grid.best_params_)
print("Best ROC AUC Score:", rf_grid.best_score_)

# Train final model with best parameters
final_model = RandomForestClassifier(**rf_grid.best_params_, random_state=42)
final_model.fit(X_train_scaled, y_train)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': final_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('Top 10 Most Important Features')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Final model evaluation
y_pred_final = final_model.predict(X_test_scaled)
final_accuracy = accuracy_score(y_test, y_pred_final)
final_roc_auc = roc_auc_score(y_test, final_model.predict_proba(X_test_scaled)[:, 1])

print("\nFinal Model Performance:")
print(f"Accuracy: {final_accuracy:.4f}")
print(f"ROC AUC: {final_roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_final))

# Save the model pipeline components
model_info = {
    'model': final_model,
    'scaler': scaler,
    'original_features': original_features,
    'encoded_features': encoded_features,
    'feature_importance': feature_importance.to_dict()
}

joblib.dump(model_info, 'heart_disease_model_pipeline.joblib')

print("\nModel pipeline has been saved to disk.")

def prepare_input_data(data):
    """
    Prepare input data for prediction by applying the same preprocessing steps
    """
    # Ensure all original features are present
    missing_cols = set(original_features) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing features in input data: {missing_cols}")

    # Convert to DataFrame if not already
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    # Handle missing values
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = SimpleImputer(strategy='median').fit_transform(data[numeric_cols])

    # Apply one-hot encoding
    data_encoded = pd.get_dummies(data, drop_first=True)

    # Ensure all encoded features are present
    for col in encoded_features:
        if col not in data_encoded.columns:
            data_encoded[col] = 0

    # Ensure correct column order
    data_encoded = data_encoded[encoded_features]

    return data_encoded

def predict_heart_disease_risk(data):
    """
    Make predictions on new data.
    data should be a DataFrame with the same features as the training data
    """
    try:
        # Prepare the input data
        processed_data = prepare_input_data(data)
        
        # Scale the features
        data_scaled = scaler.transform(processed_data)
        
        # Make prediction
        prediction = final_model.predict_proba(data_scaled)[:, 1]
        return prediction
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise

print("\nExample of using the prediction function:")
# Create a sample input
sample_input = X_test.iloc[0:1].copy()
risk_probability = predict_heart_disease_risk(sample_input)
print(f"Predicted heart disease risk probability: {risk_probability[0]:.4f}")

# Print feature names for reference
print("\nRequired features for prediction:")
for feature in original_features:
    print(feature) 