#!/bin/bash
## 01 scripts explanation
# ABCD : name of dataset
# sex : name of task
# divfreqBERT : name of model (step : 2)
# seed1 : seed is set as 1. this decides splits

## 02 environment setting
# conda activate {your environment}
# cd {your directory which contains 'main.py'}

python main.py --dataset_name ABCD --step 2 --batch_size_phase2 32 --lr_init_phase2 3e-5 --workers_phase2 8 \
--fine_tune_task binary_classification --target ADHD_label --intermediate_vec 180 --fmri_type divided_timeseries \
--nEpochs_phase2 100 --filtering_type Boxcar --transformer_hidden_layers 8 --num_heads 12 --exp_name finetune \
--seed 1 --sequence_length_phase2 348 --divide_by_lorentzian --seq_part head --use_raw_knee \
--fmri_dividing_type three_channels --use_high_freq \
--spatiotemporal --spat_diff_loss_type minus_log \
--spatial_loss_factor 1.0 --finetune \
--pretrained_model_weights_path '{your model weight path here}' \
--prepare_visualization

## if you use --prepare_visualization, then test phase will not computed and model with best acc/auroc will be stored in experiment folder.