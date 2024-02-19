#!/bin/bash

set -e

cd src/01-module

# Run the feature pipeline
python -m iris_feature_pipeline.py

# Run the batch inference pipeline
python -m iris_batch_inference_pipeline.py

# jupyter nbconvert --to notebook --execute iris-feature-pipeline.ipynb
# jupyter nbconvert --to notebook --execute iris-batch-inference-pipeline.ipynb

