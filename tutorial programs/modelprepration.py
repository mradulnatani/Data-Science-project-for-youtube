import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# Function to load and preprocess the bank dataset
def load_and_prepare_data(file_path):
    """
    Load and preprocess the bank dataset.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """
    print("Loading data...")
    bank = pd.read_csv(file_path)
    print(f"Data successfully loaded with shape {bank.shape}.")
    
    # Add an index column
    bank['index'] = pd.Series(range(bank.shape[0]))
    
    # Replace '999' in 'previous' with NaN
    bank['previous'] = bank['previous'].replace({999: np.nan})
    
    # Convert education to numeric values
    dict_edu = {"primary": 0, "secondary": 1, "tertiary": 2, "unknown": np.nan}
    bank['education_numeric'] = bank['education'].replace(dict_edu)
    
    # Add z-score for age
    bank['age_z'] = stats.zscore(bank['age'])
    
    # Replace outliers in age
    bank_outliers = bank.query('age_z > 3 | age_z < -3')
    print(f"Identified {len(bank_outliers)} outliers in the 'age' column.")
    
    return bank

# Function to partition the data into training and testing sets
def partition_data(bank, test_size=0.25, random_state=7):
    """
    Partition the data into training and testing sets.
    
    Args:
        bank (pd.DataFrame): Input dataframe.
        test_size (float): Proportion of test set.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        tuple: Training and testing dataframes.
    """
    print("\nPartitioning data into training and testing sets...")
    bank_train, bank_test = train_test_split(bank, test_size=test_size, random_state=random_state)
    print(f"Training dataset shape: {bank_train.shape}")
    print(f"Testing dataset shape: {bank_test.shape}")
    return bank_train, bank_test

# Function to balance the training data by resampling the minority class
def balance_training_data(bank_train, target_column='y', desired_proportion=0.3):
    """
    Balance the training data by resampling the minority class.
    
    Args:
        bank_train (pd.DataFrame): Training dataframe.
        target_column (str): Name of the target variable column.
        desired_proportion (float): Desired proportion of minority class.
    
    Returns:
        pd.DataFrame: Balanced training dataframe.
    """
    print("\nBalancing training data...")
    minority_class = bank_train[bank_train[target_column] == 'yes']
    majority_class = bank_train[bank_train[target_column] == 'no']
    
    total_records = len(bank_train)
    current_minority_count = len(minority_class)
    x = int((desired_proportion * total_records - current_minority_count) / (1 - desired_proportion))
    
    resampled_minority = minority_class.sample(n=x, replace=True)
    bank_train_rebal = pd.concat([bank_train, resampled_minority])
    
    print("Original training data response distribution:")
    print(bank_train[target_column].value_counts(normalize=True))
    print("\nRebalanced training data response distribution:")
    print(bank_train_rebal[target_column].value_counts(normalize=True))
    
    return bank_train_rebal

# Function to prepare features and target for model training
def prepare_features(bank, target_column='y'):
    """
    Prepares features and target for model training.
    """
    # Define features (X) and target (y)
    X = bank.drop(columns=[target_column])
    y = bank[target_column].map({'yes': 1, 'no': 0})  # Convert target to binary
    
    # Define preprocessing for numerical and categorical features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[ 
        ('imputer', SimpleImputer(strategy='mean')), 
        ('scaler', StandardScaler()) 
    ]) 

    categorical_transformer = Pipeline(steps=[ 
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('onehot', OneHotEncoder(handle_unknown='ignore')) 
    ]) 

    preprocessor = ColumnTransformer(
        transformers=[ 
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features) 
        ]
    )
    
    return X, y, preprocessor

# Function to train the model and evaluate it
def train_model(X, y, preprocessor):
    """
    Train a logistic regression model.
    """
    print("\nTraining model...")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Create pipeline with preprocessing and model
    pipeline = Pipeline(steps=[ 
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression()) 
    ]) 

    # Train the model
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return pipeline

# Main function to run the entire process
def main():
    # File path to the dataset
    file_path = r"D:\GIT\pythondev\1\bank.csv"
    
    # Load the data
    print("Loading data...")
    bank = load_and_prepare_data(file_path)
    
    # Partition the data
    print("\nPartitioning data into training and testing sets...")
    bank_train, bank_test = partition_data(bank)

    # Balance the training data
    print("\nBalancing training data...")
    target_column = 'y'  # Updated target column name
    bank_train_rebal = balance_training_data(bank_train, target_column=target_column)
    
    # Prepare features
    print("\nPreparing features...")
    X, y, preprocessor = prepare_features(bank_train_rebal, target_column=target_column)
    
    # Train and evaluate model
    train_model(X, y, preprocessor)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred during execution: {e}")
