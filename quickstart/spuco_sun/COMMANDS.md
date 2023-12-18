# Commands 

# Queues 

for i in {0..7}; do guild run queue -b --gpus="$i" -y; done

# ERM 

guild run spuco_sun_erm.py root_dir=/data/spucosun/5.0 arch='[cliprn50]' lr='[1e-3, 1e-4, 1e-5]' weight_decay='[1e-4, 5e-4, 1e-2, 1e-1, 1.0]' pretrained=True --stage-trials

arch=resnet50 lr=1e-5 weight_decay=1e-4

arch=cliprn50 lr=1e-5 weight_decay=1e-1

# GB 

guild run spuco_sun_gb.py root_dir=/data/spucosun/5.0 arch='[cliprn50]' lr='[1e-3, 1e-4, 1e-5]' weight_decay='[1e-4, 5e-4, 1e-2, 1e-1, 1.0]' pretrained=True --stage-trials

arch=resnet50 lr=1e-4 weight_decay=1e-4

arch=cliprn50 lr=1e-3 weight_decay=1e-4

# GDRO

guild run spuco_sun_groupdro.py root_dir=/data/spucosun/5.0 lr='[1e-3, 1e-4, 1e-5]' weight_decay='[1e-4, 5e-4, 1e-2, 1e-1, 1.0]' num_epochs='[40,80]' arch='[cliprn50, resnet50]' pretrained=True --stage-trials

# SSA 

guild run ssa.py root_dir=/data/spucosun/6.0 lr=1e-4 weight_decay=1e-4 infer_lr=1e-4 infer_weight_decay=1e-4 arch=resnet50 pretrained=True --stage-trials

# DFR

guild run dfr.py root_dir=/data/spucosun/6.0 lr=1e-4 weight_decay=1e-4 arch=resnet50 pretrained=True --stage-trials

# LRMix

guild run lrmix.py root_dir=/data/spucosun/6.0 lr=1e-4 weight_decay=1e-4 arch=resnet50 pretrained=True --stage-trials

# CNC 

guild run cnc.py root_dir=/data/spucosun/6.0 inf_lr=1e-4 inf_weight_decay=1e-4 arch=resnet50 pretrained=True --stage-trials

# JTT 

guild run jtt.py root_dir=/data/spucosun/6.0 lr=1e-4 weight_decay=1e-4 arch=resnet50 pretrained=True --stage-trials

# EIIL - run by Yu

guild run spuco_sun_eiil root_dir=/data/spucosun/5.0 erm_lr=1e-4 erm_weight_decay=1e-4 gdro_lr=1e-5 gdro_weight_decay=1e-2 infer_num_epochs='[1,2,3,4,5,6,7,8,9,10,20,30,40]' eiil_num_steps='[10000, 20000]' eiil_lr='[0.01, 0.001]' arch=resnet50 pretrained=True --stage-trials

# SPARE - run by Yu

guild run root_dir=/data/spucosun/5.0 spuco_sun_spare erm_lr=1e-4 erm_weight_decay=1e-4 lr='[1e-3, 1e-4, 1e-5]' weight_decay='[1e-4, 5e-4, 1e-2, 1e-1, 1.0]' infer_num_epochs='[1,2,3]' num_clusters='[2,5]' high_sampling_power='[2,3]' arch=resnet50 pretrained=True --stage-trials
