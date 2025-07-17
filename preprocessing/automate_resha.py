import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess(df):
    numerical_cols_with_missing = df.select_dtypes(include=np.number).columns[df.select_dtypes(include=np.number).isnull().any()]
    categorical_cols_with_missing = df.select_dtypes(include='object').columns[df.select_dtypes(include='object').isnull().any()]

    if len(numerical_cols_with_missing) > 0:
        imputer_numerical = SimpleImputer(strategy='median')
        df[numerical_cols_with_missing] = imputer_numerical.fit_transform(
            df[numerical_cols_with_missing]
        )

    if len(categorical_cols_with_missing) > 0:
        imputer_categorical = SimpleImputer(strategy='most_frequent')
        df[categorical_cols_with_missing] = imputer_categorical.fit_transform(
            df[categorical_cols_with_missing]
        )

    categorical_cols = df.select_dtypes(include='object').columns

    # Apply Label Encoding
    label_encoders = {}
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])

    numerical_cols = df.select_dtypes(include=np.number).columns
    cols_to_scale = [col for col in numerical_cols if col != 'Personality']

    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    return train_df, test_df

if __name__ == "__main__":
    df = pd.read_csv("personality_dataset.csv")
    train_df, test_df = preprocess(df)

    if not os.path.exists('./preprocessing/preprocessed'):
        os.makedirs('./preprocessing/preprocessed')

    train_df.to_csv('./preprocessing/preprocessed/train_df.csv', index=False)
    test_df.to_csv('./preprocessing/preprocessed/test_df.csv', index=False)
