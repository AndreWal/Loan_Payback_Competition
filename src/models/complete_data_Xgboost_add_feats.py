import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import mlflow
    import optuna
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
    from sklearn.metrics import confusion_matrix, accuracy_score
    return StratifiedKFold, XGBClassifier, cross_val_score, mlflow, optuna, pd


@app.cell
def _(mlflow):
    mlflow.set_experiment("loan_payback_rf")
    return


@app.cell
def _(pd):
    fold = "data/processed/Data_feature_training_add.csv"
    startdat = pd.read_csv(fold)

    sdate = 20112025

    startdat.head()
    return sdate, startdat


@app.cell
def _(startdat):
    train_x = startdat.drop(["loan_paid_back"], axis=1)
    train_y = startdat["loan_paid_back"]
    return train_x, train_y


@app.cell
def _(train_y):
    pos_frac = train_y.mean()
    neg_frac = 1 - pos_frac
    base_spw = neg_frac / pos_frac
    return (base_spw,)


@app.cell
def _(
    StratifiedKFold,
    XGBClassifier,
    base_spw,
    cross_val_score,
    mlflow,
    sdate,
    train_x,
    train_y,
):
    def objective(trial):
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}") as child_run:
            params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
                    "learning_rate": trial.suggest_float("learning_rate",
                                                         1e-4, 0.1, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),

                    # class imbalance
                    "scale_pos_weight": trial.suggest_float(
                        "scale_pos_weight",
                        0.5 * base_spw,
                        2.0 * base_spw,
                    ),

                    "objective": "binary:logistic",
                    "eval_metric": "auc",
                    "tree_method": "hist",
                    "random_state": sdate,
                    "n_jobs": -1,
                }

            mlflow.log_params(params)

            xgb = XGBClassifier(**params)

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=sdate)

            scores = cross_val_score(
                xgb,
                train_x,
                train_y,
                cv=cv,
                scoring="roc_auc",
                n_jobs=1,  
            )

            mean_roc = scores.mean()

            mlflow.log_metrics({"mean_roc": mean_roc})

            trial.set_user_attr("mean_roc", mean_roc)
            return mean_roc
    return (objective,)


@app.cell
def _(mlflow, objective, optuna):
    with mlflow.start_run(run_name="study") as run:
        # Log the experiment settings
        n_trials = 25
        mlflow.log_param("n_trials", n_trials)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        # Log the best trial and its run ID
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metrics({"best_roc": study.best_value})
        if best_run_id := study.best_trial.user_attrs.get("run_id"):
            mlflow.log_param("best_child_run_id", best_run_id)
    return (study,)


@app.cell
def _(study):
    best_params = study.best_trial.params

    print(best_params)
    return (best_params,)


@app.cell
def _(XGBClassifier, best_params, train_x, train_y):
    xgb = XGBClassifier(**best_params)

    xgb.fit(train_x, train_y)
    return


@app.cell
def _(cat, pd):
    tardat = "data/processed/Data_feature_test_add.csv"
    targetdat = pd.read_csv(tardat)

    ids = targetdat["id"].copy()

    X_test = targetdat.drop(columns=["id"])

    probs = cat.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame({"id": ids, "loan_paid_back": probs})

    submission.to_csv("data/processed/Xgb_submission_complete_add.csv", index=False)
    return


if __name__ == "__main__":
    app.run()
