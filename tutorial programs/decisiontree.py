import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

def prepare_data(file_path):
    """
    Load and preprocess the dataset for decision tree and random forest modeling
    """
    # Load data
    data = pd.read_csv(file_path)
    
    # Rename columns if necessary
    data.columns = [col.replace(' ', '_') for col in data.columns]
    
    # Encode categorical variables
    le = LabelEncoder()
    data['Marital_status'] = le.fit_transform(data['Marital_status'])
    data['Income'] = le.fit_transform(data['Income'])
    
    return data

def build_decision_tree(X, y, method='gini', max_leaf_nodes=5):
    """
    Build a decision tree classifier using CART or Entropy criterion
    
    Parameters:
    - X: predictor variables
    - y: target variable
    - method: 'gini' or 'entropy'
    - max_leaf_nodes: maximum number of leaf nodes
    
    Returns:
    - Fitted decision tree classifier
    """
    dt = DecisionTreeClassifier(
        criterion=method, 
        max_leaf_nodes=max_leaf_nodes
    )
    dt.fit(X, y)
    return dt

def build_random_forest(X, y, n_trees=100, method='gini'):
    """
    Build a random forest classifier
    
    Parameters:
    - X: predictor variables
    - y: target variable
    - n_trees: number of trees in the forest
    - method: splitting criterion
    
    Returns:
    - Fitted random forest classifier
    """
    rf = RandomForestClassifier(
        n_estimators=n_trees, 
        criterion=method
    )
    rf.fit(X, y)
    return rf

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    Parameters:
    - model: trained classifier
    - X_test: test predictor variables
    - y_test: test target variable
    
    Returns:
    - Accuracy score and classification report
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return accuracy, report

def main():
    # Load and prepare data
    data_path = 'D:\GIT\pythondev\1\bank.csv'  # Update with your file path
    data = prepare_data(data_path)
    
    # Separate features and target
    X = data[['Marital_status', 'Cap_Gains_Losses']]
    y = data['Income']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # CART Decision Tree (Gini criterion)
    cart_gini = build_decision_tree(X_train, y_train, method='gini')
    cart_gini_acc, cart_gini_report = evaluate_model(cart_gini, X_test, y_test)
    
    print("CART Decision Tree (Gini) Results:")
    print(f"Accuracy: {cart_gini_acc}")
    print("Classification Report:\n", cart_gini_report)
    
    # C5.0-like Decision Tree (Entropy criterion)
    c50_tree = build_decision_tree(X_train, y_train, method='entropy')
    c50_tree_acc, c50_tree_report = evaluate_model(c50_tree, X_test, y_test)
    
    print("\nC5.0-like Decision Tree (Entropy) Results:")
    print(f"Accuracy: {c50_tree_acc}")
    print("Classification Report:\n", c50_tree_report)
    
    # Random Forest
    random_forest = build_random_forest(X_train, y_train)
    rf_acc, rf_report = evaluate_model(random_forest, X_test, y_test)
    
    print("\nRandom Forest Results:")
    print(f"Accuracy: {rf_acc}")
    print("Classification Report:\n", rf_report)
    
    # Optional: Export decision tree visualization
    export_graphviz(cart_gini, 
                    out_file='cart_decision_tree.dot', 
                    feature_names=['Marital_status', 'Cap_Gains_Losses'],
                    class_names=['<=50K', '>50K'])

if __name__ == '__main__':
    main()