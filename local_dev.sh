#!/usr/bin/env bash

if [ ! -d .venv ]; then
  uv venv .venv
fi

uv pip install -r requirements.txt

source .venv/bin/activate