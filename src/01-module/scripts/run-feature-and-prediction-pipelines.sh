#!/bin/bash

set -e

cd src/01-module

# Run the feature pipeline
python iris-feature-pipeline.py

# Run the batch inference pipeline
python iris_batch_inference_pipeline.py

# jupyter nbconvert --to notebook --execute iris-feature-pipeline.ipynb
# jupyter nbconvert --to notebook --execute iris-batch-inference-pipeline.ipynb
 