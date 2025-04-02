import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score, classification_report

import warnings
warnings.filterwarnings('ignore')

#Load the dataset
file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
data = pd.read_csv(file_path)
print(data.head())

#Exploratory Data Analysis

# Distribution of target variable
# Ensure the target column exists and rename if necessary
if 'NObeyesdad' not in data.columns:
    print("Renaming target column to 'NObeyesdad'.")
    data.rename(columns={data.columns[-1]: 'NObeyesdad'}, inplace=True)

sns.countplot(y='NObeyesdad', data=data)
plt.title('Distribution of Obesity Levels')
plt.show()

#Missing Values in Dataset
print("Missing Values in Dataset:\n", data.isnull().sum())

# Display dataset info (data types, non-null values)
print("\nDataset Information:\n")
print(data.info())

# Display summary statistics
print("\nDataset Summary Statistics:\n")
print(data.describe())

#Preprocessing the data
# Standardizing continuous numerical features
continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[continuous_columns])
scaled_df = pd.DataFrame(scaled_features, columns=continuous_columns)

# Converting to a DataFrame
scaled_df = pd.DataFrame(scaled_features, columns=continuous_columns)

# Combining with the original dataset
scaled_data = pd.concat([data.drop(columns=continuous_columns), scaled_df], axis=1)

#One-hot encoding
# Identifying categorical columns
categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('NObeyesdad')  # Exclude target column

# Applying one-hot encoding
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(scaled_data[categorical_columns])

# Converting to a DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

# Combining with the original dataset
prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)

#Encode the target variable
# Encoding the target variable
prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes
prepped_data.head()

#Separate the input and target data
# Preparing final dataset
X = prepped_data.drop('NObeyesdad', axis=1)
y = prepped_data['NObeyesdad']
# Display dataset shapes
print(f"Shape of X (features): {X.shape}")
print(f"Shape of y (target): {y.shape}")

#Model training and evaluation

#Splitting the data set
# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Display dataset shapes
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

#Logistic Regression with One-vs-All
model_ova = LogisticRegression(multi_class='ovr', max_iter=2000, solver='lbfgs')

model_ova.fit(X_train, y_train)
# Check model performance on training data
train_accuracy = model_ova.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy:.4f}")

# Check model performance on test data
test_accuracy = model_ova.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predictions
y_pred_ova = model_ova.predict(X_test)

# Evaluation metrics for OvA
print("One-vs-All (OvA) Strategy")
print(f"Accuracy: {accuracy_score(y_test, y_pred_ova) * 100:.2f}%")
# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred_ova))


#Logistic Regression with OvO

# Training logistic regression model using One-vs-One
model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
model_ovo.fit(X_train, y_train)

# Predictions
y_pred_ovo = model_ovo.predict(X_test)

# Evaluation metrics for OvO
print("One-vs-One (OvO) Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ovo),2)}%")


#Task 1: Experiment with different test sizes in the train_test_split method (e.g., 0.1, 0.3) and observe the impact on model performance.

test_sizes = [0.1, 0.3]

for test_size in test_sizes:
    # Splitting Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Training Model (One-vs-All)
    model_ova.fit(X_train, y_train)
    
    # Predictions
    y_pred = model_ova.predict(X_test)
    
    # Display Results
    print("="*50)
    print(f"Test Size: {test_size}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
#Task 2: Plot a bar chart of feature importance using the coefficients from the One vs All logistic regression model. Also try for the One vs One model.

# Compute feature importance for OvA (One-vs-All)
feature_importance_ova = np.mean(np.abs(model_ova.coef_), axis=0)

# Compute feature importance for OvO (One-vs-One)
feature_importance_ovo = np.mean(
    np.abs([estimator.coef_ for estimator in model_ovo.estimators_]), axis=(0, 1)
)

# Sorting by importance
sorted_idx = np.argsort(feature_importance_ova)

# Plot OvA vs. OvO
plt.barh(prepped_data.drop('NObeyesdad', axis=1).columns[sorted_idx], feature_importance_ova[sorted_idx], label="OvA", alpha=0.6, color="blue")
plt.barh(prepped_data.drop('NObeyesdad', axis=1).columns[sorted_idx], feature_importance_ovo[sorted_idx], label="OvO", alpha=0.6, color="orange")
plt.barh(X.columns[sorted_idx], feature_importance_ovo[sorted_idx], label="OvO", alpha=0.6, color="orange")

plt.title("Feature Importance (OvA vs. OvO)")
plt.xlabel("Importance")
plt.legend()
plt.show()

#TASK 3:  Write a function obesity_risk_pipeline to automate the entire pipeline: 
def obesity_risk_pipeline(data_path, test_size=0.2):
    """
    Automates the pipeline for obesity risk prediction.
    
    Steps:
    1. Load and preprocess data (standardization + one-hot encoding)
    2. Train Logistic Regression models using One-vs-All (OvA) and One-vs-One (OvO)
    3. Evaluate models and visualize feature importance
    
    Parameters:
        data_path (str): Path to the dataset
        test_size (float): Proportion of data to be used for testing
    """
    # Load data
    data = pd.read_csv(data_path)

    # Standardizing continuous numerical features
    continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[continuous_columns])
    scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))
    scaled_data = pd.concat([data.drop(columns=continuous_columns), scaled_df], axis=1)

    # Identifying categorical columns
    categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
    categorical_columns.remove('NObeyesdad')  # Exclude target column

    # Applying one-hot encoding
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_features = encoder.fit_transform(scaled_data[categorical_columns])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))
    prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)

    # Encoding the target variable
    prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes

    # Splitting data
    X = prepped_data.drop('NObeyesdad', axis=1)
    y = prepped_data['NObeyesdad']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    # One-vs-All (OvA) Model
    model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
    model_ova.fit(X_train, y_train)
    y_pred_ova = model_ova.predict(X_test)
    print(f"OvA Accuracy: {np.round(100 * accuracy_score(y_test, y_pred_ova), 2)}%")

    # One-vs-One (OvO) Model
    model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
    model_ovo.fit(X_train, y_train)
    y_pred_ovo = model_ovo.predict(X_test)
    print(f"OvO Accuracy: {np.round(100 * accuracy_score(y_test, y_pred_ovo), 2)}%")

    # Feature Importance (OvA)
    feature_importance = np.mean(np.abs(model_ova.coef_), axis=0)
    sorted_idx = np.argsort(feature_importance)

    plt.barh(prepped_data.drop('NObeyesdad', axis=1).columns[sorted_idx], feature_importance[sorted_idx], color="blue", alpha=0.7)
    plt.barh(X.columns[sorted_idx], feature_importance[sorted_idx], color="blue", alpha=0.7)
    plt.title("Feature Importance (One-vs-All)")
    plt.xlabel("Importance")
    plt.show()

# Run the pipeline
file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
try:
    obesity_risk_pipeline(file_path, test_size=0.2)
except Exception as e:
    print(f"An error occurred: {e}")