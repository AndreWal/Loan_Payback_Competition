import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import mlflow
    import optuna
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    return (
        RandomForestClassifier,
        cross_val_score,
        mlflow,
        optuna,
        pd,
        train_test_split,
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
def _(startdat, train_test_split):
    train, test = train_test_split(startdat, test_size=0.3, random_state=19112025)
    train_x = train.drop(["loan_paid_back"], axis=1)
    train_y = train["loan_paid_back"]

    test_x = test.drop(["loan_paid_back"], axis=1)
    test_y = test["loan_paid_back"]
    return train_x, train_y


@app.cell
def _(RandomForestClassifier, cross_val_score, mlflow, train_x, train_y):
    def objective(trial):
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}") as child_run:
            params = {
                "n_estimators": trial.suggest_int("rf_n_estimators", 100, 800, step=50),
                "max_depth": trial.suggest_categorical(
                    "rf_max_depth", [None, 4, 8, 16, 32, 64]
                ),
                "max_features": trial.suggest_categorical(
                    "rf_max_features", ["sqrt", "log2", None, 0.3, 0.5, 0.7, 1.0]
                ),
                "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 20),
                "max_leaf_nodes": trial.suggest_categorical(
                    "rf_max_leaf_nodes", [None, 16, 32, 64, 128]
                ),
                "bootstrap": trial.suggest_categorical("rf_bootstrap", [True, False]),
                "class_weight": trial.suggest_categorical(
                    "rf_class_weight", [None, "balanced"]
                ),
            }

            mlflow.log_params(params)

            rfc = RandomForestClassifier(**params, n_jobs=15)

            scores = cross_val_score(
                rfc,
                train_x,
                train_y,
                cv=5,
                scoring="f1",
                n_jobs=5,  
            )

            mean_f1 = scores.mean()

            mlflow.log_metrics({"f1": mean_f1})

            trial.set_user_attr("f1_mean", mean_f1)
            return mean_f1
    return (objective,)


@app.cell
def _(mlflow, objective, optuna):
    with mlflow.start_run(run_name="study") as run:
        # Log the experiment settings
        n_trials = 30
        mlflow.log_param("n_trials", n_trials)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        # Log the best trial and its run ID
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metrics({"best_f1": study.best_value})
        if best_run_id := study.best_trial.user_attrs.get("run_id"):
            mlflow.log_param("best_child_run_id", best_run_id)
    return


if __name__ == "__main__":
    app.run()
