import pandas as pd

DROP_COLUMNS = ['search_id', 'feature_0', 'feature_3', 'feature_77', 'feature_53', 'feature_12']

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(DROP_COLUMNS, axis=1)