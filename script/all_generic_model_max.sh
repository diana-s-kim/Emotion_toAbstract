#!/bin/bash
python -u train.py --img_dir "/ibex/ai/home/kimds/Research/P2/data/wikiart_resize/" --version "pre_processed_v1" --data "max_emotion" --model "resnet34_orig" --feature_level "last" --save_model_dir "all_generic_model_max/"
