import sys
import pandas as pd
from typing import NoReturn
from preprocess import drop_columns
from xgboost import XGBClassifier


def predict(path_to_data: str) -> NoReturn:
    df = pd.read_csv(path_to_data)
    df = drop_columns(df)
    model = XGBClassifier()
    model.load_model('./model.xgb')
    print(model.predict(df))

if __name__ == '__main__':
    path_to_data = sys.argv[1]
    predict(path_to_data)