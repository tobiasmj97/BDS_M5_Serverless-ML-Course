#!/bin/bash

set -e

cd src/02-module

# Run the feature pipeline
python 2_cc_feature_pipeline.py

#jupyter nbconvert --to notebook --execute 2_cc_feature_pipeline.ipynb

