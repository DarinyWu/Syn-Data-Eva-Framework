#---DarinyWu--2025.07.15---#
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

def evaluate_model_based_regression_utility_cv(real_df, synth_df, features, target_col, categorical_features, n_splits=5, random_state=42):
    """
    Evaluate synthetic data utility using GradientBoostingRegressor with k-fold cross-validation.
    Trains on real and synthetic data, evaluates on real test folds.

    Parameters:
    - real_df: real dataset
    - synth_df: synthetic dataset
    - features: list of feature columns
    - target_col: continuous target column (e.g., end_time)
    - categorical_features: list of categorical feature names
    - n_splits: number of CV folds
    - random_state: random seed

    Returns:
    - dict with average MAE and RMSE for both real-trained and synth-trained models
    """
    numerical_features = [f for f in features if f not in categorical_features]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
    ])

    X_real = real_df[features]
    y_real = real_df[target_col]
    X_synth = synth_df[features]
    y_synth = synth_df[target_col]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    mae_real, rmse_real = [], []
    mae_synth, rmse_synth = [], []

    for train_idx, test_idx in kf.split(X_real):
        X_train_real, X_test_real = X_real.iloc[train_idx], X_real.iloc[test_idx]
        y_train_real, y_test_real = y_real.iloc[train_idx], y_real.iloc[test_idx]

        # Real-trained pipeline
        pipe_real = Pipeline([
            ('preprocess', preprocessor),
            ('reg', GradientBoostingRegressor(random_state=random_state))
        ])
        pipe_real.fit(X_train_real, y_train_real)
        y_pred_real = pipe_real.predict(X_test_real)
        mae_real.append(mean_absolute_error(y_test_real, y_pred_real))
        rmse_real.append(np.sqrt(mean_squared_error(y_test_real, y_pred_real)))

        # Synth-trained pipeline
        pipe_synth = Pipeline([
            ('preprocess', preprocessor),
            ('reg', GradientBoostingRegressor(random_state=random_state))
        ])
        pipe_synth.fit(X_synth, y_synth)
        y_pred_synth = pipe_synth.predict(X_test_real)
        mae_synth.append(mean_absolute_error(y_test_real, y_pred_synth))
        rmse_synth.append(np.sqrt(mean_squared_error(y_test_real, y_pred_synth)))

    return {
        'Real→Real MAE (CV mean)': np.mean(mae_real),
        'Real→Real RMSE (CV mean)': np.mean(rmse_real),
        'Synth→Real MAE (CV mean)': np.mean(mae_synth),
        'Synth→Real RMSE (CV mean)': np.mean(rmse_synth),
    }

#Customize your Usage
