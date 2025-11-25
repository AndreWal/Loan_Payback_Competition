import pandas as pd

cat = pd.read_csv("data/processed/Cat_submission_complete_add.csv")

xgb = pd.read_csv("data/processed/Xgb_submission_complete_add.csv")

lgbm = pd.read_csv("data/processed/LGBM_submission_complete_add.csv")

preds = (cat["loan_paid_back"] + xgb["loan_paid_back"] + lgbm["loan_paid_back"]) / 3

submission = pd.DataFrame({"id": cat["id"], "loan_paid_back": preds})

submission.to_csv("data/processed/avg_submission_complete.csv", index=False)