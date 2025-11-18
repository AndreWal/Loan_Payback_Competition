import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    return np, pd


@app.cell
def _(np, pd):
    def load_data(file_path: str) -> pd.DataFrame:
        """Load data from a CSV file into a pandas DataFrame."""
        return pd.read_csv(file_path)

    def transform(data) -> pd.DataFrame:
        mapping = {"Master's": "Master", "Bachelor's": "Bachelor"}
        data["education_level"] = data["education_level"].replace(mapping)
        data["ln_income"] = np.log(data["annual_income"])
        data["ln_debt"] = np.log(data["debt_to_income_ratio"])
        cols = ["gender","marital_status","education_level","employment_status","loan_purpose"]
        data = pd.get_dummies(data, columns=cols, drop_first=True)
        drop = ["annual_income","debt_to_income_ratio"]
        data.drop(columns=drop, inplace=True)
        return data
    return load_data, transform


@app.cell
def _(load_data):
    ##### Training data
    train = load_data("data/raw/train.csv")

    train.head()
    return (train,)


@app.cell
def _(train, transform):
    train_feat = transform(train)

    train_feat.head()
    return (train_feat,)


@app.cell
def _(train_feat):
    train_feat.to_csv("data/processed/Data_feature_training.csv")
    return


@app.cell
def _(load_data):
    ##### Test data
    test = load_data("data/raw/test.csv")

    test.head()
    return (test,)


@app.cell
def _(test, transform):
    test_feat = transform(test)

    test_feat.head()
    return (test_feat,)


@app.cell
def _(test_feat):
    test_feat.to_csv("data/processed/Data_feature_test.csv")
    return


if __name__ == "__main__":
    app.run()
