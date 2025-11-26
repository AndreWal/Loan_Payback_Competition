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

    def transform(data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # 1) Clean / normalize simple categoricals
        if "education_level" in df.columns:
            # Make sure Master's / Bachelor's are written consistently
            mapping = {"Master's": "Master", "Bachelor's": "Bachelor"}
            df["education_level"] = df["education_level"].replace(mapping)

        # 2) Core numeric building blocks
        income = df["annual_income"].clip(lower=1.0)
        dti = df["debt_to_income_ratio"].clip(lower=1e-6)
        loan_amt = df["loan_amount"]
        rate = df["interest_rate"]
        rate_dec = rate / 100.0
        df["interest_rate_dec"] = rate_dec  # interest rate in decimals

        # 3) Affordability / stress features
        # Loan as share of income (original feature name)
        df["loan_sh"] = loan_amt / income
        # Non-linear version
        df["loan_sh_sq"] = df["loan_sh"] ** 2

        # Existing debt & disposable income
        df["annual_existing_debt"] = income * dti
        df["annual_disposable_income"] = income - df["annual_existing_debt"]
        disp = df["annual_disposable_income"].clip(lower=1.0)

        # Interest cost and ratios
        df["annual_interest_cost"] = loan_amt * rate_dec
        df["interest_to_income"] = df["annual_interest_cost"] / income
        df["loan_to_disposable_income"] = loan_amt / disp
        df["interest_to_disposable"] = df["annual_interest_cost"] / disp

        # 4) Logs and simple non-linearities
        df["ln_income"] = np.log(income)
        df["ln_debt"] = np.log(dti)
        df["ln_loan_amount"] = np.log(loan_amt.clip(lower=1.0))
        df["ln_disposable_income"] = np.log(disp)

        # DTI curvature + thresholds (non-linear “cliff” effects)
        df["dti_sq"] = dti ** 2
        df["dti_high_025"] = (dti > 0.25).astype(int)
        df["dti_high_040"] = (dti > 0.40).astype(int)

        # Simple interaction between price and leverage
        df["rate_dti_product"] = rate_dec * dti

        # 5) Credit score curvature
        if "credit_score" in df.columns:
            df["credit_score_sq"] = df["credit_score"] ** 2

        # 6) Grade / pricing structure
        if "grade_subgrade" in df.columns:
            # e.g. 'C3' -> 'C', 3
            df["grade_letter"] = df["grade_subgrade"].str[0]
            df["grade_num"] = df["grade_subgrade"].str[1:].astype(int)

            grade_map = {"A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1}
            df["grade_ord"] = df["grade_letter"].map(grade_map)

            # Cohort-level mean rate and deviation ("over/underpriced" loans)
            grp = df.groupby("grade_subgrade")["interest_rate"]
            df["grade_mean_rate"] = grp.transform("mean")
            df["grade_rate_diff"] = df["interest_rate"] - df["grade_mean_rate"]

            # Interactions between grade and affordability
            df["loan_sh_x_grade_ord"] = df["loan_sh"] * df["grade_ord"]
            df["loan_to_disp_x_grade_ord"] = df["loan_to_disposable_income"] * df["grade_ord"]
        else:
            # If for some reason grade_subgrade is missing, keep columns but fill with NaN
            df["grade_letter"] = np.nan
            df["grade_num"] = np.nan
            df["grade_ord"] = np.nan
            df["grade_mean_rate"] = np.nan
            df["grade_rate_diff"] = np.nan
            df["loan_sh_x_grade_ord"] = np.nan
            df["loan_to_disp_x_grade_ord"] = np.nan

        # 7) One-hot categorical variables (including the new grade_letter)
        cat_cols = []
        for col in ["gender",
                    "marital_status",
                    "education_level",
                    "employment_status",
                    "loan_purpose",
                    "grade_letter"]:
            if col in df.columns:
                cat_cols.append(col)

        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, dtype=int)

        # 8) Drop raw grade_subgrade (we’ve encoded it)
        if "grade_subgrade" in df.columns:
            df = df.drop(columns=["grade_subgrade"])

        return df
    return load_data, transform


@app.cell
def _(load_data, pd):
    ##### Training data
    addtrain = load_data("data/raw/loan_dataset_20000.csv")

    orgtrain = load_data("data/raw/train.csv")

    add_red = addtrain[addtrain.columns.intersection(orgtrain.columns)]

    # now you can stack / append / concatenate
    train = pd.concat([orgtrain, add_red], ignore_index=True)

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
    train_feat.to_csv("data/processed/Data_feature_origin_data.csv", index=False)
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
