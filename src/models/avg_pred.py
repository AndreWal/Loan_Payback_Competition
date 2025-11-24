import pandas as pd

cat = pd.read_csv("data/processed/Cat_submission_complete_add.csv")

rf = pd.read_csv("data/processed/RF_submission_complete.csv")

preds = (cat["loan_paid_back"] + rf["loan_paid_back"]) / 2

submission = pd.DataFrame({"id": cat["id"], "loan_paid_back": preds})

submission.to_csv("data/processed/avg_submission_complete.csv", index=False)