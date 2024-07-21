#!/bin/bash

## 01 scripts explanation
# 1-1. you can change 'communicability_option'
## if communicability_option==remove_high_comm_node, node with high communicability will be removed and model will learn to fill high-communicable nodes.
## elif communicability_option==remove_low_comm_node, node with low communicability will be removed and model will learn to fill low-communicable nodes.
# 1-2. you can change 'num_hub_ROIs' - number of nodes that you remove.
## num_hub_ROIs === 380 for Schaefer 400 atlas
## num_hub_ROIs == 170 for HCPMMP1 atlas

## 02 environment setting
# conda activate {your environment}
# cd {your directory which contains 'main.py'}

python main.py --dataset_name UKB --step 4 --batch_size_phase4 32 --lr_init_phase4 3e-5  \
--workers_phase4 16 --intermediate_vec 400 --fmri_type divided_timeseries --target reconstruction \
--nEpochs_phase4 1000 --filtering_type Boxcar --transformer_hidden_layers 8 --num_heads 12 \
--exp_name pretraining_seed1 \
--seed 1 --sequence_length_phase4 464 --divide_by_lorentzian --seq_part head --use_raw_knee \
--fmri_dividing_type three_channels --use_high_freq --compute_pink --norm_location post --norm_type batch \
--use_mask_loss --masking_method spatiotemporal --spatial_masking_type hub_ROIs --num_hub_ROIs 380 --communicability_option remove_high_comm_node \
--temporal_masking_type time_window --temporal_masking_window_size 20 --window_interval_rate 2 --spat_diff_loss_type minus_log \
--spatiotemporal --spatial_loss_factor 0.1 