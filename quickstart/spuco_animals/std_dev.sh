#!/bin/bash

# Commands with seed=2
tmux new-session -d -s eiil_seed2 "python spuco_animals_eiil.py --erm_lr=1e-3 --erm_weight_decay=1e-3 --gdro_lr=1e-4 --gdro_weight_decay=1e-1 --infer_num_epochs=2 --eiil_num_steps=10000 --eiil_lr=1e-3 --arch=cliprn50 --only_train_projection --seed=2 --gpu=1"
tmux new-session -d -s erm_seed2 "python spuco_animals_erm.py --lr=1e-3 --weight_decay=1e-3 --arch=cliprn50 --only_train_projection --seed=2 --gpu=2"
tmux new-session -d -s group_balance_seed2 "python spuco_animals_group_balance.py --lr=1e-3 --weight_decay=1e-3 --arch=cliprn50 --only_train_projection --seed=2 --gpu=4"
tmux new-session -d -s groupdro_seed2 "python spuco_animals_groupdro.py --lr=1e-4 --weight_decay=1e-4 --arch=cliprn50 --only_train_projection --seed=2 --gpu=1"
tmux new-session -d -s pde_seed2 "python spuco_animals_pde.py --lr=1e-3 --weight_decay=1e-3 --warmup_epochs=20 --expansion_size=10 --expansion_interval=2 --arch=cliprn50 --only_train_projection --pretrained --seed=2 --gpu=2"
tmux new-session -d -s spare_seed2 "python spuco_animals_spare.py --erm_lr=1e-3 --erm_weight_decay=1e-4 --lr=1e-4 --weight_decay=1e-2 --infer_num_epochs=2 --num_clusters=2 --high_sampling_power=2 --arch=cliprn50 --only_train_projection --pretrained --seed=2 --gpu=4"
tmux new-session -d -s ssa_seed2 "python spuco_animals_ssa.py --infer_lr=1e-4 --infer_weight_decay=1e-3 --infer_num_iters=1000 --lr=1e-3 --weight_decay=1e-4 --num_epochs=40 --arch=cliprn50 --only_train_projection --pretrained --seed=2 --gpu=1"

# Commands with seed=3
tmux new-session -d -s eiil_seed3 "python spuco_animals_eiil.py --erm_lr=1e-3 --erm_weight_decay=1e-3 --gdro_lr=1e-4 --gdro_weight_decay=1e-1 --infer_num_epochs=2 --eiil_num_steps=10000 --eiil_lr=1e-3 --arch=cliprn50 --only_train_projection --seed=3 --gpu=2"
tmux new-session -d -s erm_seed3 "python spuco_animals_erm.py --lr=1e-3 --weight_decay=1e-3 --arch=cliprn50 --only_train_projection --seed=3 --gpu=4"
tmux new-session -d -s group_balance_seed3 "python spuco_animals_group_balance.py --lr=1e-3 --weight_decay=1e-3 --arch=cliprn50 --only_train_projection --seed=3 --gpu=1"
tmux new-session -d -s groupdro_seed3 "python spuco_animals_groupdro.py --lr=1e-4 --weight_decay=1e-4 --arch=cliprn50 --only_train_projection --seed=3 --gpu=2"
tmux new-session -d -s pde_seed3 "python spuco_animals_pde.py --lr=1e-3 --weight_decay=1e-3 --warmup_epochs=20 --expansion_size=10 --expansion_interval=2 --arch=cliprn50 --only_train_projection --pretrained --seed=3 --gpu=4"
tmux new-session -d -s spare_seed3 "python spuco_animals_spare.py --erm_lr=1e-3 --erm_weight_decay=1e-4 --lr=1e-4 --weight_decay=1e-2 --infer_num_epochs=2 --num_clusters=2 --high_sampling_power=2 --arch=cliprn50 --only_train_projection --pretrained --seed=3 --gpu=1"
tmux new-session -d -s ssa_seed3 "python spuco_animals_ssa.py --infer_lr=1e-4 --infer_weight_decay=1e-3 --infer_num_iters=1000 --lr=1e-3 --weight_decay=1e-4 --num_epochs=40 --arch=cliprn50 --only_train_projection --pretrained --seed=3 --gpu=2"

echo "All commands have been started in separate tmux sessions."
