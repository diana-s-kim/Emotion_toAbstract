#!/bin/bash
python -u train.py --img_dir "/ibex/ai/home/kimds/Research/P2/data/wikiart_resize/" --version "pre_processed_v1" --data "hist_emotion" --model "resnet34_gram_transformer" --feature_level "conv4" --feature_set "texture" "composition" "color" --save_model_dir "all_gram_transformer_model/"
