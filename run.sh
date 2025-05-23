#!/bin/bash
# Simple helper to install dependencies and run training
pip install -r requirements.txt
python train.py "$@"
