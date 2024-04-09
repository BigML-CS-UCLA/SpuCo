#!/bin/bash

tmux new-session -d -s session_lr1e5_wd1e5 'python spuco_animals_groupdro.py --arch cliprn50 --only_train_projection --lr 1e-5 --weight_decay 1e-5 --pretrained --gpu 0'
tmux new-session -d -s session_lr1e5_wd1e4 'python spuco_animals_groupdro.py --arch cliprn50 --only_train_projection --lr 1e-5 --weight_decay 1e-4 --pretrained --gpu 1'
tmux new-session -d -s session_lr1e5_wd1e3 'python spuco_animals_groupdro.py --arch cliprn50 --only_train_projection --lr 1e-5 --weight_decay 1e-3 --pretrained --gpu 2'
tmux new-session -d -s session_lr1e4_wd1e5 'python spuco_animals_groupdro.py --arch cliprn50 --only_train_projection --lr 1e-4 --weight_decay 1e-5 --pretrained --gpu 3'
tmux new-session -d -s session_lr1e4_wd1e4 'python spuco_animals_groupdro.py --arch cliprn50 --only_train_projection --lr 1e-4 --weight_decay 1e-4 --pretrained --gpu 4'
tmux new-session -d -s session_lr1e4_wd1e3 'python spuco_animals_groupdro.py --arch cliprn50 --only_train_projection --lr 1e-4 --weight_decay 1e-3 --pretrained --gpu 5'
tmux new-session -d -s session_lr1e3_wd1e5 'python spuco_animals_groupdro.py --arch cliprn50 --only_train_projection --lr 1e-3 --weight_decay 1e-5 --pretrained --gpu 0'
tmux new-session -d -s session_lr1e3_wd1e4 'python spuco_animals_groupdro.py --arch cliprn50 --only_train_projection --lr 1e-3 --weight_decay 1e-4 --pretrained --gpu 1'
tmux new-session -d -s session_lr1e3_wd1e3 'python spuco_animals_groupdro.py --arch cliprn50 --only_train_projection --lr 1e-3 --weight_decay 1e-3 --pretrained --gpu 2'
tmux new-session -d -s session_lr1e3_wd1e2 'python spuco_animals_groupdro.py --arch cliprn50 --only_train_projection --lr 1e-3 --weight_decay 1e-2 --pretrained --gpu 3'
tmux new-session -d -s session_lr1e3_wd1e1 'python spuco_animals_groupdro.py --arch cliprn50 --only_train_projection --lr 1e-3 --weight_decay 1e-1 --pretrained --gpu 4'
