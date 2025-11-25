# Create basic files
echo "Creating key files..."
echo "# config file" > configs/config.yaml
echo "# Logging yaml" > configs/logging.yaml
touch .env

cat > .gitignore << 'EOF'
# --- Python cache / artifacts ---
__pycache__/
*.py[cod]
*.pyo
*.pyd
*.so

# --- Virtual environments ---
.venv/
venv/
.env/
*.venv/

# --- Jupyter / notebooks ---
.ipynb_checkpoints/
*.ipynb_snapshots/

# --- Data (Kaggle datasets, features, predictions, etc.) ---
data/

# --- Logs / temporary files ---
*.log
logs/
*.tmp
*.swp
*.swo

# --- MLflow / experiment artifacts ---
mlruns/
mlartifacts/

# --- VS Code / editor settings ---
.vscode/
*.code-workspace

# --- OS / misc ---
catboost_info/
.DS_Store
Thumbs.db
EOF


cat > requirements.txt << 'REQ'
scikit-learn
pandas
optuna
marimo
matplotlib
mlflow
catboost
xgboost
lightgbm
REQ