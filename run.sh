#!/bin/bash
# Simple helper to install dependencies and run training
pip install -r requirements.txt
export OMP_NUM_THREADS=8
python train.py "$@"
