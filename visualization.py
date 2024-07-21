import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

import os
from tqdm import tqdm
import json
import numpy as np

from pathlib import Path

import sys

from model import *
import argparse
from trainer import *
from data_preprocess_and_load.dataloaders import *


## ABIDE!

def get_arguments(base_path = os.getcwd()):
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str,default="baseline") 
    parser.add_argument('--dataset_name', type=str, choices=['HCP1200', 'ABCD', 'ABIDE', 'UKB'], default="ABCD")
    parser.add_argument('--gaussian_blur', action='store_true') 
    parser.add_argument('--fmri_type', type=str, choices=['timeseries', 'frequency', 'divided_timeseries', 'divided_frequency', 'time_domain_low', 'time_domain_ultralow', 'time_domain_high' , 'frequency_domain_low', 'frequency_domain_ultralow', 'timeseries_and_frequency'], default="timeseries")
    parser.add_argument('--intermediate_vec', type=int, choices=[84, 48, 22, 180, 200, 400, 246], default=180)
    parser.add_argument('--shaefer_num_networks', type=int, choices=[7, 17], default=17)
    parser.add_argument('--fmri_timeseries_path', default='/storage/bigdata/ABCD/fmriprep/1.rs_fmri/5.ROI_DATA') ## labserver
    parser.add_argument('--ukb_path', default='/scratch/connectome/stellasybae/UKB_ROI') ## labserver
    parser.add_argument('--abide_path', default='/global/cfs/projectdirs/m4244/stella/ABIDE/3.ROI_DATA') ## perlmutter
    parser.add_argument('--base_path', default=base_path)
    parser.add_argument('--step', default='1', choices=['1','2','3', '4', '5'], help='which step you want to run')
    
    
    parser.add_argument('--target', type=str, default='sex')
    #, choices=['sex','age','ASD_label','ADHD_label','nihtbx_totalcomp_uncorrected','nihtbx_fluidcomp_uncorrected', 'ADHD_label_robust', 'SuicideIdeationtoAttempt', 'BMI', 'ASD', 'ASD_label', 'reconstruction', 'phq9_score'],help='fine_tune_task must be specified as follows -- {sex:classification, age:regression, ASD_label (ABCD) or ASD (ABIDE) :classification, ADHD_label:classification, nihtbx_***:regression}')
    parser.add_argument('--fine_tune_task',
                        choices=['regression','binary_classification'],
                        help='fine tune model objective. choose binary_classification in case of a binary classification task')
    parser.add_argument('--label_scaling_method', default = 'standardization', choices=['standardization', 'minmax'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--visualization', action='store_true')
    parser.add_argument('--norm_axis', default=1, type=int, choices=[0,1,None])
    
    parser.add_argument('--cuda', default=True)
    parser.add_argument('--log_dir', type=str, default=os.path.join(base_path, 'runs'))
    parser.add_argument('--transformer_hidden_layers', type=int,default=8)
    parser.add_argument('--train_split', default=0.7)
    parser.add_argument('--val_split', default=0.15)
    parser.add_argument('--running_mean_size', default=5000)
    
    # DDP configs:
    parser.add_argument('--world_size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--local_rank', default=-1, type=int, 
                        help='local rank for distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--init_method', default='file', type=str, choices=['file','env'], help='DDP init method')
    parser.add_argument('--non_distributed', action='store_true')
    parser.add_argument('--distributed', default=False)

    # AMP configs:
    parser.add_argument('--amp', action='store_false')
    parser.add_argument('--gradient_clipping', action='store_true')
    parser.add_argument('--clip_max_norm', type=float, default=1.0)
    
    # Gradient accumulation
    parser.add_argument("--accumulation_steps", default=1, type=int,required=False,help='mini batch size == accumulation_steps * args.train_batch_size')
    
    # Nsight profiling
    parser.add_argument("--profiling", action='store_true')
    
    # multimodality options
    parser.add_argument('--fmri_multimodality_type', default='three_channels', choices=['cross_attention','two_channels', 'three_channels'])
    parser.add_argument('--compute_pink', action='store_true')
    
    #wandb related
    parser.add_argument('--wandb_key', default='108101f4b9c3e31a235aa58307d1c6b548cfb54a', type=str,  help='default: key for Stella')
    parser.add_argument('--wandb_mode', default='online', type=str,  help='online|offline')
    
    
    ## divfreqBERT
    # Model Options
    parser.add_argument('--filtering_type', default='Boxcar', choices=['FIR', 'Boxcar'])
    parser.add_argument('--use_high_freq', action='store_true')
    parser.add_argument('--embedding_integration_method', type=str, choices=['concat','attention', 'average', 'gap'], default='average')
    parser.add_argument('--parameter_sharing', action='store_true') # temporal
    parser.add_argument('--parameter_sharing_for_spatial_attention', action='store_true') # spatial
    parser.add_argument('--no_parameter_sharing_for_classifier', action='store_true')
    parser.add_argument('--dropout_on_output', action='store_true')
    parser.add_argument('--norm_location', default='post', choices=['pre', 'post'])
    parser.add_argument('--norm_type', default='batch', choices=['batch', 'group', 'layer', 'instance'])                   

    
    # dividing rationale
    parser.add_argument('--divide_by_lorentzian', action='store_true')
    parser.add_argument('--use_raw_knee', action='store_true')
    parser.add_argument('--seq_part', type=str, default='tail')
    parser.add_argument('--use_three_channels', action='store_true')
    parser.add_argument('--use_spatial_attention_map', action='store_true')
    
    # Dropouts
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='attention dropout')
    parser.add_argument('--attn_dropout_u', type=float, default=0.0,
                        help='attention dropout (for ultralow)')
    parser.add_argument('--relu_dropout', type=float, default=0.1,
                        help='relu dropout')
    parser.add_argument('--embed_dropout', type=float, default=0.25,
                        help='embedding dropout')
    parser.add_argument('--res_dropout', type=float, default=0.1,
                        help='residual block dropout')
    parser.add_argument('--out_dropout', type=float, default=0.0,
                        help='output layer dropout')

    # Architecture
    parser.add_argument('--nlevels', type=int, default=12,
                        help='number of layers in the network (default: 12)')
    parser.add_argument('--num_heads_mult', type=int, default=12,
                        help='number of heads for the mutlimodal transformer network (default: 12)')
    parser.add_argument('--num_heads_spatial', type=int, default=12,
                        help='number of heads for the spatial transformer network (default: 12)')
    parser.add_argument('--attn_mask', action='store_false',
                        help='use attention mask for Transformer (default: true)')
    
    
    ## for finetune
    parser.add_argument('--pretrained_model_weights_path', default=None)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--finetune_test', action='store_true', help='test phase of finetuning task')
    
    
    ##phase 3 (test)
    ## spatiotemporal
    parser.add_argument('--spatial', action = 'store_true') # for phase 1 
    parser.add_argument('--spatiotemporal', action = 'store_true') # for phase 1 
    parser.add_argument('--spatiotemporal_type', type=str, default='spatial_attention', choices=['another_stream', 'matrix_multiplication', 'spatial_attention'])
    parser.add_argument('--spat_diff_loss_type', type=str, default='minus_log', choices=['minus_log', 'reciprocal_log', 'exp_minus', 'log_loss', 'exp_whole'])
    parser.add_argument('--spatial_loss_factor', type=float, default=0.1)
    
    parser.add_argument('--transformer_dropout_rate', type=float, default=0.3)
    parser.add_argument('--task', type=str, default='test')
    parser.add_argument('--model_weights_path_phase2', default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--nEpochs', type=int, default=20)
    parser.add_argument('--augment_prob', default=0)
    parser.add_argument('--optim', default='AdamW')
    parser.add_argument('--weight_decay3', type=float, default=1e-2)
    parser.add_argument('--lr_policy', default='step', help='learning rate policy: step|SGDR')
    parser.add_argument('--lr_init', type=float, default=1e-4)
    parser.add_argument('--lr_gamma', type=float, default=0.9)
    parser.add_argument('--lr_step', type=int, default=1500)
    parser.add_argument('--lr_warmup', type=int, default=100)
    parser.add_argument('--sequence_length', type=int,default=348)
    parser.add_argument('--workers', type=int, default=4)

    parser.add_argument('--model_path', type=str)
    parser.add_argument('--save_dir', type=str)
    
    args = parser.parse_args()
        
    return args


args = get_arguments()
model_path = args.model_path
model = Transformer_Finetune_Three_Channels(**vars(args))
state_dict = torch.load(model_path)['model_state_dict']
# pos_ids = model.transformer.bert.embeddings.position_ids
#del state_dict['transformer.bert.embeddings.position_ids']
model.load_state_dict(state_dict)
model.eval()
model.cuda(0) if torch.cuda.is_available() else model

def get_activation(dict, name):
    def hook(model, input, output):
        dict[name] = output[0].detach().tolist()
    return hook


integrated_gradients = IntegratedGradients(model)
noise_tunnel = NoiseTunnel(integrated_gradients)

data_handler = DataHandler(**vars(args))
_, _, test_loader = data_handler.create_dataloaders()

save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)
dataset_name = args.dataset_name
kwargs = {
    "nt_samples": 5,
    "nt_samples_batch_size": 5,
    "nt_type": "smoothgrad_sq", # 1
    #"stdevs": 0.05,
    "internal_batch_size": 5,
}


for idx, data in enumerate(tqdm(test_loader),0):
    subj_name = data['subject_name'][0]
    
    # input itself
    input_low = data['fmri_lowfreq_sequence'].float().requires_grad_(True).cuda(0)
    input_ultralow = data['fmri_ultralowfreq_sequence'].requires_grad_(True).float().cuda(0)
    input_high = data['fmri_highfreq_sequence'].float().requires_grad_(True).cuda(0)
   
    # intermediate modules
    # att_mat_high = model.high_spatial_attention(input_high.permute(0, 2, 1)).float().cuda(0)
    # att_mat_low = model.low_spatial_attention(input_low.permute(0, 2, 1)).float().cuda(0)
    # att_mat_ultralow = model.ultralow_spatial_attention(input_ultralow.permute(0, 2, 1)).float().cuda(0)
    
    label = data[args.target].float().cuda(0)
    pred = model(input_high, input_low, input_ultralow)[args.fine_tune_task]
    pred_prob = torch.sigmoid(pred)
    pred_int = (pred_prob>0.5).int().item()
    target_int = label.int().item()
    
    #only choose corrected samples
    
    if pred_int == target_int:
        if target_int == 0:
            if pred_prob <= 0.25:
                # target 0
                file_dir = os.path.join(save_dir, f'{dataset_name}_target0')
                os.makedirs(file_dir,exist_ok=True)
                
                # 01 filename
                file_path_input_low = os.path.join(file_dir, f"{subj_name}_input_low.pt")
                file_path_input_ultralow = os.path.join(file_dir, f"{subj_name}_input_ultralow.pt")
                file_path_input_high = os.path.join(file_dir, f"{subj_name}_input_high.pt")
                activation_path = os.path.join(file_dir, f"{subj_name}_att_mat_activation.json")
                gradient_path = os.path.join(file_dir, f"{subj_name}_att_mat_gradient.json")

                # 02-2 noise tunnel - att mat
                # 특정 레이어에 forward hook 등록
                activation_layer_high = 'high_spatial_attention'  # 중간 값과 그래디언트를 추출할 레이어 1
                activation_layer_low = 'low_spatial_attention'  # 중간 값과 그래디언트를 추출할 레이어 2 
                activation_layer_ultralow = 'ultralow_spatial_attention'  # 중간 값과 그래디언트를 추출할 레이어 3

                activations = {}
                model.high_spatial_attention.register_forward_hook(get_activation(activations, activation_layer_high))
                model.low_spatial_attention.register_forward_hook(get_activation(activations, activation_layer_low))
                model.ultralow_spatial_attention.register_forward_hook(get_activation(activations, activation_layer_ultralow))

                # 특정 레이어에 backward hook 등록
                gradients = {}
                model.high_spatial_attention.register_backward_hook(get_activation(gradients, activation_layer_high))
                model.low_spatial_attention.register_backward_hook(get_activation(gradients, activation_layer_low))
                model.ultralow_spatial_attention.register_backward_hook(get_activation(gradients, activation_layer_ultralow))

                # 순전파 수행
                h = model.high_spatial_attention(input_high.permute(0, 2, 1)).float().cuda(0)
                l = model.low_spatial_attention(input_low.permute(0, 2, 1)).float().cuda(0)
                u = model.ultralow_spatial_attention(input_ultralow.permute(0, 2, 1)).float().cuda(0)

                # 역전파 수행
                loss_fn = nn.L1Loss()
                spat_diff_loss = -torch.log((loss_fn(h, l)+loss_fn(h, u)+loss_fn(l, u)))
                spat_diff_loss.backward()
                
                
                with open(activation_path, 'w') as f : 
                    json.dump(activations, f, indent=4)
                with open(gradient_path, 'w') as f : 
                    json.dump(gradients, f, indent=4)
                        
                print(f'saving {subj_name}')
        
        elif target_int == 1:
            if pred_prob >= 0.75:
                # target 1
                file_dir = os.path.join(save_dir, f'{dataset_name}_target1')
                os.makedirs(file_dir,exist_ok=True)
                # 01 filename
                file_path_input_low = os.path.join(file_dir, f"{subj_name}_input_low.pt")
                file_path_input_ultralow = os.path.join(file_dir, f"{subj_name}_input_ultralow.pt")
                file_path_input_high = os.path.join(file_dir, f"{subj_name}_input_high.pt")
                activation_path = os.path.join(file_dir, f"{subj_name}_att_mat_activation.json")
                gradient_path = os.path.join(file_dir, f"{subj_name}_att_mat_gradient.json")

                # 02-2 noise tunnel - att mat
                # 특정 레이어에 forward hook 등록
                activation_layer_high = 'high_spatial_attention'  # 중간 값과 그래디언트를 추출할 레이어 1
                activation_layer_low = 'low_spatial_attention'  # 중간 값과 그래디언트를 추출할 레이어 2 
                activation_layer_ultralow = 'ultralow_spatial_attention'  # 중간 값과 그래디언트를 추출할 레이어 3

                activations = {}
                model.high_spatial_attention.register_forward_hook(get_activation(activations, activation_layer_high))
                model.low_spatial_attention.register_forward_hook(get_activation(activations, activation_layer_low))
                model.ultralow_spatial_attention.register_forward_hook(get_activation(activations, activation_layer_ultralow))

                # 특정 레이어에 backward hook 등록
                gradients = {}
                model.high_spatial_attention.register_backward_hook(get_activation(gradients, activation_layer_high))
                model.low_spatial_attention.register_backward_hook(get_activation(gradients, activation_layer_low))
                model.ultralow_spatial_attention.register_backward_hook(get_activation(gradients, activation_layer_ultralow))

                # 순전파 수행
                h = model.high_spatial_attention(input_high.permute(0, 2, 1)).float().cuda(0)
                l = model.low_spatial_attention(input_low.permute(0, 2, 1)).float().cuda(0)
                u = model.ultralow_spatial_attention(input_ultralow.permute(0, 2, 1)).float().cuda(0)

                # 역전파 수행
                loss_fn = nn.L1Loss()
                spat_diff_loss = -torch.log((loss_fn(h, l)+loss_fn(h, u)+loss_fn(l, u)))
                spat_diff_loss.backward()
                
                # activation = activations[activation_layer]
                # gradient = gradients[activation_layer]
                
                with open(activation_path, 'w') as f : 
                    json.dump(activations, f, indent=4)
                with open(gradient_path, 'w') as f : 
                    json.dump(gradients, f, indent=4)
                
                print(f'saving {subj_name}')