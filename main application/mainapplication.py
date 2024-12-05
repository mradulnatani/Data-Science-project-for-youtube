import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def load_and_prepare_data(file_path):
    """
    Load and preprocess the bank dataset.
    """
    print("Loading data...")
    bank = pd.read_csv(file_path)
    print(f"Data successfully loaded with shape {bank.shape}.")
    
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

def balance_training_data(bank_train, target_column='y', desired_proportion=0.3):
    """
    Balance the training data by resampling the minority class.
    """
    print("\nBalancing training data...")
    minority_class = bank_train[bank_train[target_column] == 'yes']
    majority_class = bank_train[bank_train[target_column] == 'no']
    
    total_records = len(bank_train)
    current_minority_count = len(minority_class)
    x = int((desired_proportion * total_records - current_minority_count) / (1 - desired_proportion))
    
    resampled_minority = minority_class.sample(n=x, replace=True, random_state=42)
    bank_train_rebal = pd.concat([bank_train, resampled_minority])
    
    print("Original training data response distribution:")
    print(bank_train[target_column].value_counts(normalize=True))
    print("\nRebalanced training data response distribution:")
    print(bank_train_rebal[target_column].value_counts(normalize=True))
    
    return bank_train_rebal

def prepare_features(bank, target_column='y'):
    """
    Prepares features and target for model training.
    """
    X = bank.drop(columns=[target_column])
    y = bank[target_column].map({'yes': 1, 'no': 0})
    
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

def train_decision_tree(X, y, preprocessor, method='gini', max_leaf_nodes=10):
    """
    Train a decision tree classifier.
    """
    print("\nTraining Decision Tree model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(criterion=method, max_leaf_nodes=max_leaf_nodes, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return pipeline, X_test, y_test

def train_naive_bayes(X, y, preprocessor):
    """
    Train a Naïve Bayes classifier.
    """
    print("\nTraining Naïve Bayes model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GaussianNB())
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return pipeline, X_test, y_test

def export_decision_tree(pipeline, feature_names, output_file='decision_tree.dot'):
    """
    Export the trained decision tree model to a .dot file for visualization.
    """
    tree_model = pipeline.named_steps['classifier']
    
    # Get the full list of feature names from the preprocessor
    preprocessor = pipeline.named_steps['preprocessor']
    onehot_features = preprocessor.transformers_[1][1]['onehot'].get_feature_names_out()
    all_feature_names = list(feature_names[:len(preprocessor.transformers_[0][2])]) + list(onehot_features)
    
    # Export the tree to a .dot file
    export_graphviz(
        tree_model,
        out_file=output_file,
        feature_names=all_feature_names,
        class_names=['No', 'Yes'],
        filled=True,
        rounded=True,
        special_characters=True
    )
    print(f"Decision tree exported to {output_file}.")

def model_evaluation(pipeline, X_test, y_test):
    """
    Evaluate the model on the test set.
    """
    print("\nEvaluating model performance...")
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

def main():
    file_path = r"D:\GIT\pythondev\1\bank.csv"
    
    # Load and prepare data
    bank = load_and_prepare_data(file_path)
    
    # Partition data
    bank_train, _ = train_test_split(bank, test_size=0.25, random_state=42)
    
    # Balance training data
    bank_train_rebal = balance_training_data(bank_train)
    
    # Prepare features
    X, y, preprocessor = prepare_features(bank_train_rebal)
    
    # Train Decision Tree
    print("\n--- Decision Tree Evaluation ---")
    pipeline_dt, X_test, y_test = train_decision_tree(X, y, preprocessor)
    model_evaluation(pipeline_dt, X_test, y_test)
    
    # Train Naïve Bayes
    print("\n--- Naïve Bayes Evaluation ---")
    pipeline_nb, X_test, y_test = train_naive_bayes(X, y, preprocessor)
    model_evaluation(pipeline_nb, X_test, y_test)
    
    # Export Decision Tree
    export_decision_tree(pipeline_dt, X_test.columns)

if __name__ == "__main__":
    main()
