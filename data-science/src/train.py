# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using the training dataset and evaluates using the test dataset. Saves trained model.
"""

import mlflow
import argparse
import os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from matplotlib import pyplot as plt

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument('--n_estimators', type=int, default='gini',
                        help='The function to measure the quality of a split')
    parser.add_argument('--max_depth', type=int, default=None,
                        help='The maximum depth of the tree. If None, then nodes are expanded until all the leaves contain less than min_samples_split samples.')

    args = parser.parse_args()

    return args

def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''

    # Read train and test data from CSV
    train_df = pd.read_csv(Path(args.train_data)/"train.csv")
    test_df = pd.read_csv(Path(args.test_data)/"test.csv")

    # Split the data into input(X) and output(y)
    y_train = train_df.iloc[:, -1]
    X_train = train_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]

    # One-hot encode categorical columns
    X_train = pd.get_dummies(X_train)
    X_test  = pd.get_dummies(X_test)

    # Align test data with training columns
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # Initialize and train a Random forest regressor
    model = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Log model hyperparameters
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    # Predict using the Decision Tree Model on test data
    yhat_test = model.predict(X_test)

    # Compute and log mean_squared_error
    mse = mean_squared_error(y_test, yhat_test)
    r2 = r2_score(y_test, yhat_test)
    mae = mean_absolute_error(y_test, yhat_test)
    
    print(f'Mean Squared Error: {mse:.2f}')
    # Logging the R2 score as a metric
    mlflow.log_metric("MSE", float(mse))
    print(f'R2 Score: {r2:.2f}')
    print(f'Mean Absolute Error: {mae:.2f}')
    # Save the model
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)

if __name__ == "__main__":
    
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Test dataset input path: {args.test_data}",
        f"Model output path: {args.model_output}",
        f"n_estimators: {args.n_estimators}",
        f"Max Depth: {args.max_depth}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
