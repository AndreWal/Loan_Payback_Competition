#!/usr/bin/env bash

# Creates project directory structure for the loan payback competition

set -e
set -u

dirs=(
  "Docker"
  "data/raw"
  "data/processed"
  "data/plots"
  "notebooks/exploratory"
  "notebooks/modelling"
  "src/data"
  "src/features"
  "src/models"
  "src/utils"
  "src/inference"
  "configs"
  "tests"
)

echo "Creating directories..."
for d in "${dirs[@]}"; do
  mkdir -p "$d"
  echo "  Created: $d"
done

echo "Folder structure setup complete."
