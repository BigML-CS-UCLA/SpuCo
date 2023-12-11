# Commands 

# ERM 

guild run spuco_sun_erm.py wandb=True wandb_entity=spuco root_dir=/data/spucosun/3.0 arch='[resnet50, cliprn50]' lr='[1e-3, 1e-4, 1e-5]' weight_decay='[1e-4, 5e-4, 1e-2, 1e-1, 1.0]' pretrained=True --stage-trials

# GB 

guild run spuco_sun_gb.py wandb=True wandb_entity=spuco root_dir='[/data/spucosun/3.0]' arch='[resnet50, cliprn50]' lr='[1e-3, 1e-4, 1e-5]' weight_decay='[1e-4, 5e-4, 1e-2, 1e-1, 1.0]' pretrained=True --stage-trials