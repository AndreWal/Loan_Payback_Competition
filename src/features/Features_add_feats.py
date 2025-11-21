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
        data["loan_sh"] = data["loan_amount"] / data["annual_income"] 
        data["loan_int"] = data["loan_amount"] * data["interest_rate"]
        data["ln_income"] = np.log(data["annual_income"])
        data["ln_debt"] = np.log(data["debt_to_income_ratio"])
        rate_dec = data["interest_rate"] / 100.0
        data["annual_interest_cost"] = data["loan_amount"] * rate_dec
        data["interest_to_income"] = data["annual_interest_cost"] / data["annual_income"]
        data["rate_income_product"] = data["interest_rate"] * data["debt_to_income_ratio"]
        data["annual_existing_debt"] = data["annual_income"] * data["debt_to_income_ratio"]
        data["annual_disposable_income"] = data["annual_income"] - data["annual_existing_debt"]
        data["loan_to_disposable_income"] = data["loan_amount"] / data["annual_disposable_income"].clip(lower=1.0)
        data["grade_letter"] = data["grade_subgrade"].str[0]
        data["grade_num"] = data["grade_subgrade"].str[1:].astype(int)
        grade_map = {"A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1}
        data["grade_ord"] = data["grade_letter"].map(grade_map)
        rate_dec = data["interest_rate"] / 100.0
        data["interest_to_disposable"] = (data["loan_amount"] * rate_dec) / data["annual_disposable_income"].clip(lower=1.0)
        data.drop(["grade_subgrade"], axis=1, inplace=True)
        cols = ["gender","marital_status","education_level","employment_status","loan_purpose","grade_letter"]
        data = pd.get_dummies(data, columns=cols, dtype=int)
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
    return (train_feat,)


@app.cell
def _(train_feat):
    train_feat.drop(columns=["id"], inplace=True)
    return


@app.cell
def _(train_feat):
    train_feat.to_csv("data/processed/Data_feature_training_add.csv", index=False)
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
    test_feat.to_csv("data/processed/Data_feature_test_add.csv", index=False)
    return


if __name__ == "__main__":
    app.run()
