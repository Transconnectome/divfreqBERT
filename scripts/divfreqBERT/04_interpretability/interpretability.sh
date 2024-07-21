#!/bin/bash
## 01 scripts explanation
# after finetuning or training-from-scratch, you can run visualization.py

## 02 environment setting
# conda activate {your environment}
# cd {your directory which contains 'visualization.py'}

python visualization.py --dataset_name ABCD --step 3 \
--fine_tune_task binary_classification --target ADHD_label --intermediate_vec 180 --fmri_type divided_timeseries \
--transformer_hidden_layers 8 --num_heads 12 --exp_name test --wandb_mode offline \
--transformer_dropout_rate 0.3 --seed 1 --divide_by_lorentzian --seq_part head --use_raw_knee --fmri_dividing_type three_channels \
--use_high_freq --spatiotemporal --spatial_loss_factor 1.0 --finetune \
--model_path {your finetuned model path} \
--save_dir {the path you want to store interpretability results}