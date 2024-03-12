import sys
import pandas as pd
from preprocess import drop_columns
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from typing import NoReturn


def train(path_to_data: str) -> NoReturn:
    df = pd.read_csv(path_to_data)
    df = drop_columns(df)
    X = df.drop('target', axis=1)
    y = df['target']

    majority_class = y.mode()[0]
    minority_class = 1 - majority_class
    imbalance = y.value_counts()[majority_class] / y.value_counts()[minority_class]

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01, 0.05],
        'gamma': [0, 0.1, 0.5],
        'scale_pos_weight': [imbalance]
    }

    model = XGBClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted')
    grid_search.fit(X, y)

    print(f'Best params: {grid_search.best_params_}')
    print(f'Best score: {grid_search.best_score_:.2f}')

    grid_search.best_estimator_.save_model('model.xgb')

if __name__ == '__main__':
    path_to_data = sys.argv[1]
    train(path_to_data)