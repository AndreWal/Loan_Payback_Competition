import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import mlflow
    import optuna
    from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
    return (
        RandomForestClassifier,
        StratifiedKFold,
        cross_val_score,
        mlflow,
        optuna,
        pd,
    )


@app.cell
def _(mlflow):
    # Use filesystem backend in ./mlruns (default). You can keep this implicit,
    # but it's clearer to set the experiment name explicitly:
    mlflow.set_experiment("loan_payback_rf_baseline")
    return


@app.cell
def _(pd):
    fold = "data/processed/Data_feature_training.csv"
    startdat = pd.read_csv(fold)

    startdat.head()
    return (startdat,)


@app.cell
def _(startdat):
    train_x = startdat.drop(["loan_paid_back"], axis=1)
    train_y = startdat["loan_paid_back"]

    sdate = 20112025
    return sdate, train_x, train_y


@app.cell
def _(
    RandomForestClassifier,
    StratifiedKFold,
    cross_val_score,
    mlflow,
    sdate,
    train_x,
    train_y,
):
    def objective(trial):
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}") as child_run:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
                "max_depth": trial.suggest_categorical("max_depth", [None, 8, 16, 32]),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", 0.3, 0.5, 0.7]
                ),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_leaf_nodes": trial.suggest_categorical(
                    "max_leaf_nodes", [None, 64, 128, 256]
                ),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "class_weight": trial.suggest_categorical(
                    "class_weight", [None, "balanced"]
                ),
            }

            mlflow.log_params(params)

            rfc = RandomForestClassifier(**params, n_jobs=-1, random_state=sdate)

            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=sdate)

            scores = cross_val_score(
                rfc,
                train_x,
                train_y,
                cv=cv,
                scoring="roc_auc",
                n_jobs=1,  
            )

            mean_roc = scores.mean()

            mlflow.log_metrics({"roc_auc": mean_roc})

            trial.set_user_attr("roc_auc", mean_roc)
            return mean_roc
    return (objective,)


@app.cell
def _(mlflow, objective, optuna):
    with mlflow.start_run(run_name="study") as run:
        # Log the experiment settings
        n_trials = 14
        mlflow.log_param("n_trials", n_trials)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        # Log the best trial and its run ID
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metrics({"best_acc1": study.best_value})
        if best_run_id := study.best_trial.user_attrs.get("run_id"):
            mlflow.log_param("best_child_run_id", best_run_id)
    return (study,)


@app.cell
def _(sdate, study):
    best_params = study.best_trial.params

    best_params.update({"n_jobs": -1, "random_state": sdate})

    print(best_params)
    return (best_params,)


@app.cell
def _(RandomForestClassifier, best_params, train_x, train_y):
    rfc = RandomForestClassifier(**best_params)

    rfc.fit(train_x, train_y)
    return (rfc,)


@app.cell
def _(pd, rfc):
    tardat = "data/processed/Data_feature_test.csv"
    targetdat = pd.read_csv(tardat)

    ids = targetdat["id"].copy()
    X_test = targetdat.drop(columns=["id"])

    probs = rfc.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame({"id": ids, "loan_paid_back": probs})

    submission.to_csv("data/processed/RF_submission_complete.csv", index=False)
    return


if __name__ == "__main__":
    app.run()
