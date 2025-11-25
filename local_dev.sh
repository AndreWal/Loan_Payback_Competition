#!/usr/bin/env bash

if [ ! -d .venv ]; then
  uv venv .venv
fi

uv pip install -r requirements.txt

# Activate virtual environment manually with:

source .venv/bin/activate