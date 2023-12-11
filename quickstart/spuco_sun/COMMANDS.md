# Commands 

# Queues 

for i in {0..7}; do guild run queue -b --gpus="$i" -y; done

# ERM 

guild run spuco_sun_erm.py root_dir=/data/spucosun/3.0 arch='[resnet50, cliprn50]' lr='[1e-3, 1e-4, 1e-5]' weight_decay='[1e-4, 5e-4, 1e-2, 1e-1, 1.0]' pretrained=True --stage-trials

# GB 

guild run spuco_sun_gb.py root_dir=/data/spucosun/3.0 arch='[resnet50, cliprn50]' lr='[1e-3, 1e-4, 1e-5]' weight_decay='[1e-4, 5e-4, 1e-2, 1e-1, 1.0]' pretrained=True --stage-trials

# GDRO

guild run groupdro.py root_dir=/data/spucosun/3.0 lr=‘[1e-3, 1e-4, 1e-5]’ weight_decay=‘[1e-4, 5e-4, 1e-2, 1e-1, 1.0]’ num-epochs='[40,80]' group_lr=0.01 arch=resnet50 pretrained=True --stage-trials

# SSA 

guild run ssa.py root_dir=/data/spucosun/3.0 lr='[1e-3, 1e-4, 1e-5]' weight_decay='[1e-4, 5e-4, 1e-2, 1e-1, 1.0]' infer_lr='[1e-3, 1e-4, 1e-5]' infer_weight_decay='[1e-4, 5e-4, 1e-2, 1e-1, 1.0]' arch=resnet50 pretrained=True --stage-trials

# DFR

guild run dfr.py root_dir=/data/spucosun/3.0 lr='[1e-3, 1e-4, 1e-5]' weight_decay='[1e-4, 5e-4, 1e-2, 1e-1, 1.0]' arch=resnet50 pretrained=True --stage-trials

# LRMix
guild run lrmix.py root_dir=/data/spucosun/3.0 lr='[1e-3, 1e-4, 1e-5]' weight_decay='[1e-4, 5e-4, 1e-2, 1e-1, 1.0]' arch=resnet50 pretrained=True --stage-trials

# CNC 

guild run cnc.py root_dir=/data/spucosun/3.0 inf_lr='[1e-3, 1e-4, 1e-5]' inf_weight_decay='[1e-4, 5e-4, 1e-2, 1e-1, 1.0]' arch=resnet50 pretrained=True --stage-trials

# JTT 

guild run jtt.py root_dir=/data/spucosun/3.0 lr=‘[1e-3, 1e-4, 1e-5]’ weight_decay=‘[1e-4, 5e-4, 1e-2, 1e-1, 1.0]’ arch=resnet50 pretrained=True --stage-trials

# EIIL 

guild run spuco_sun_eiil root_dir=/data/spucosun/3.0 erm_lr='[]' erm_weight_decay='[]' gdro_lr='[]' gdro_weight_decay='[]' infer_num_epochs='[1,2,3,4,5,6,7,8,9,10,20,30,40]' eiil_num_steps='[10000, 20000]' eiil_lr='[0.01, 0.001]' arch=resnet50 pretrained=True --stage-trials

# SPARE

guild run root_dir=/data/spucosun/3.0 spuco_sun_spare erm_lr='[]' erm_weight_decay='[]' lr='[1e-3, 1e-4, 1e-5]' weight_decay='[1e-4, 5e-4, 1e-2, 1e-1, 1.0]' infer_num_epochs='[1,2,3]' num_clusters='[2,5]' high_sampling_power='[2,3]' arch=resnet50 pretrained=True --stage-trials
