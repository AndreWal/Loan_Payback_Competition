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
        accuracy_score,
        confusion_matrix,
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
    mlflow.set_experiment("loan_payback_rf")
    return


@app.cell
def _(pd):
    fold = "data/processed/Data_feature_training.csv"
    startdat = pd.read_csv(fold)

    startdat.head()
    return (startdat,)


@app.cell
def _(startdat, train_test_split):
    train, test = train_test_split(
        startdat,
        test_size=0.3,
        random_state=19112025,
        stratify=startdat["loan_paid_back"],
    )

    train_x = train.drop(["loan_paid_back"], axis=1)
    train_y = train["loan_paid_back"]

    test_x = test.drop(["loan_paid_back"], axis=1)
    test_y = test["loan_paid_back"]

    test_x.shape
    return test_x, test_y, train_x, train_y


@app.cell
def _(
    RandomForestClassifier,
    StratifiedKFold,
    cross_val_score,
    mlflow,
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

            rfc = RandomForestClassifier(**params, n_jobs=-1, random_state=19112025)

            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=19112025)

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
        n_trials = 40
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
def _(study):
    best_params = study.best_trial.params

    best_params.update({"n_jobs": -1, "random_state": 19112025})

    print(best_params)
    return (best_params,)


@app.cell
def _(RandomForestClassifier, best_params, test_x, train_x, train_y):
    rfc = RandomForestClassifier(**best_params)

    rfc.fit(train_x, train_y)

    preds = rfc.predict(test_x) 
    return preds, rfc


@app.cell
def _(confusion_matrix, pd, preds, test_y):
    cm = confusion_matrix(test_y, preds)

    labels = sorted(list(set(test_y)))

    cm_rel = cm / cm.sum()   # divide by total number of samples

    cm_rel_df = pd.DataFrame(
        cm_rel,
        index=[f"true_{l}" for l in labels],
        columns=[f"pred_{l}" for l in labels]
    )

    print("Confusion matrix (relative frequencies):")
    print(cm_rel_df)
    print()
    return


@app.cell
def _(accuracy_score, preds, test_y):
    acc = accuracy_score(test_y, preds)
    print(f"Accuracy: {acc:.4f}")
    return


@app.cell
def _(pd, rfc):
    tardat = "data/processed/Data_feature_test.csv"
    targetdat = pd.read_csv(tardat)

    ids = targetdat["id"].copy()

    X_test = targetdat.drop(columns=["id"])

    probs = rfc.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame({"id": ids, "loan_paid_back": probs})

    submission.to_csv("data/processed/RF_submission_reduced.csv", index=False)
    return


if __name__ == "__main__":
    app.run()
