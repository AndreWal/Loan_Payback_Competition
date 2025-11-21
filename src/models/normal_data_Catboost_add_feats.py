import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import mlflow
    import optuna
    from catboost import CatBoostClassifier
    from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
    from sklearn.metrics import confusion_matrix, accuracy_score
    return (
        CatBoostClassifier,
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
    return test_x, test_y, train_x, train_y


@app.cell
def _(train_y):
    pos_frac = train_y.mean()
    neg_frac = 1 - pos_frac
    base_spw = neg_frac / pos_frac
    return (base_spw,)


@app.cell
def _(
    CatBoostClassifier,
    StratifiedKFold,
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
            "iterations": trial.suggest_int("iterations", 100, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
            "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5 * base_spw, 2.0 * base_spw),
            "od_type": "Iter",
            "od_wait": 20,
            "eval_metric": "AUC",         
            "loss_function": "Logloss",
                        }

            mlflow.log_params(params)

            cat = CatBoostClassifier(**params, silent=True, thread_count=-1, random_seed = sdate)

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=sdate)

            scores = cross_val_score(
                cat,
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
    return (best_params,)


@app.cell
def _(CatBoostClassifier, best_params, test_x, train_x, train_y):
    cat = CatBoostClassifier(**best_params, silent=True)

    cat.fit(train_x, train_y)

    preds = cat.predict(test_x) 
    return cat, preds


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
def _(cat, pd):
    tardat = "data/processed/Data_feature_test_add.csv"
    targetdat = pd.read_csv(tardat)

    ids = targetdat["id"].copy()

    X_test = targetdat.drop(columns=["id"])

    probs = cat.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame({"id": ids, "loan_paid_back": probs})

    submission.to_csv("data/processed/Cat_submission_reduced_add.csv", index=False)
    return


if __name__ == "__main__":
    app.run()
