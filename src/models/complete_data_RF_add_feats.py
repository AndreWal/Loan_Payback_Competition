import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import mlflow
    import optuna
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
    from sklearn.metrics import confusion_matrix, accuracy_score
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
    return


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

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=sdate)

            scores = cross_val_score(
                rfc,
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
def _(sdate, study):
    best_params = study.best_trial.params

    best_params.update({"thread_count": -1, "random_seed": sdate})

    print(best_params)
    return


@app.cell
def _(RandomForestClassifier, params, sdate, train_x, train_y):
    rfc = RandomForestClassifier(**params, n_jobs=-1, random_state=sdate)

    rfc.fit(train_x, train_y)
    return


@app.cell
def _(cat, pd):
    tardat = "data/processed/Data_feature_test_add.csv"
    targetdat = pd.read_csv(tardat)

    ids = targetdat["id"].copy()

    X_test = targetdat.drop(columns=["id"])

    probs = cat.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame({"id": ids, "loan_paid_back": probs})

    submission.to_csv("data/processed/Cat_submission_complete_add.csv", index=False)
    return


if __name__ == "__main__":
    app.run()
