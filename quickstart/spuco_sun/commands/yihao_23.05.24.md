```
guild run waterbirds_dfr.py root_dir='[/data]' lr='[1e-3,1e-4,1e-5]' weight_decay='[1e-5, 1e-4, 1e-3, 1e-2]' num_epochs='[40, 100]' arch=cliprn50 only_train_projection=True pretrained=True -m 16 --stage-trials
guild run spuco_animals_dfr.py root_dir='[/data]' lr='[1e-3,1e-4,1e-5]' weight_decay='[1e-5, 1e-4, 1e-3, 1e-2]' num_epochs='[40, 100]' arch=cliprn50 only_train_projection=True pretrained=True -m 16 --stage-trials
guild run celebA_dfr.py root_dir='[/data]' lr='[1e-3,1e-4,1e-5]' weight_decay='[1e-5, 1e-4, 1e-3, 1e-2]' num_epochs='[40, 100]' arch=cliprn50 only_train_projection=True pretrained=True -m 16 --stage-trials
guild run spuco_sun_dfr.py root_dir='[/data/spucosun/10.0, /data/spucosun/10.21]' lr='[1e-3,1e-4,1e-5]' weight_decay='[1e-5, 1e-4, 1e-3, 1e-2]' num_epochs='[40, 100]' arch=cliprn50 only_train_projection=True pretrained=True -m 16 --stage-trials

guild run waterbirds_jtt.py root_dir='[/data]' lr='[1e-3,1e-4,1e-5]' weight_decay='[1e-5, 1e-4, 1e-3, 1e-2]' num_epochs='[40, 100]' infer_num_epochs='[1,2]' arch=cliprn50 only_train_projection=True pretrained=True -m 16 --stage-trials
guild run spuco_animals_jtt.py root_dir='[/data]' lr='[1e-3,1e-4,1e-5]' weight_decay='[1e-5, 1e-4, 1e-3, 1e-2]' num_epochs='[40, 100]' infer_num_epochs='[1,2]' arch=cliprn50 only_train_projection=True pretrained=True -m 16 --stage-trials
guild run celebA_jtt.py root_dir='[/data]' lr='[1e-3,1e-4,1e-5]' weight_decay='[1e-5, 1e-4, 1e-3, 1e-2]' num_epochs='[40, 100]' infer_num_epochs='[1,2]' arch=cliprn50 only_train_projection=True pretrained=True -m 16 --stage-trials
guild run spuco_sun_jtt.py root_dir='[/data/spucosun/10.0, /data/spucosun/10.21]' lr='[1e-3,1e-4,1e-5]' weight_decay='[1e-5, 1e-4, 1e-3, 1e-2]' num_epochs='[40, 100]' infer_num_epochs='[1,2]' arch=cliprn50 only_train_projection=True pretrained=True -m 16 --stage-trials

guild run waterbirds_ssa.py root_dir='[/data]' infer_lr='[1e-2,1e-3,1e-4]' infer_weight_decay='[1e-5, 1e-4, 1e-3, 1e-2]' infer_num_iters='[1000, 45000]' lr='[1e-3,1e-4,1e-5]' weight_decay='[1e-5, 1e-4, 1e-3, 1e-2]' num_epochs='[40, 100]' arch=cliprn50 only_train_projection=True pretrained=True -m 16 --stage-trials
guild run spuco_animals_ssa.py root_dir='[/data]' infer_lr='[1e-2,1e-3,1e-4]' infer_weight_decay='[1e-5, 1e-4, 1e-3, 1e-2]' infer_num_iters='[1000, 45000]' lr='[1e-3,1e-4,1e-5]' weight_decay='[1e-5, 1e-4, 1e-3, 1e-2]' num_epochs='[40, 100]' arch=cliprn50 only_train_projection=True pretrained=True -m 16 --stage-trials
guild run celebA_ssa.py root_dir='[/data]' infer_lr='[1e-2,1e-3,1e-4]' infer_weight_decay='[1e-5, 1e-4, 1e-3, 1e-2]' infer_num_iters='[1000, 45000]' lr='[1e-3,1e-4,1e-5]' weight_decay='[1e-5, 1e-4, 1e-3, 1e-2]' num_epochs='[40, 100]' arch=cliprn50 only_train_projection=True pretrained=True -m 16 --stage-trials
guild run spuco_sun_ssa.py root_dir='[/data/spucosun/10.0, /data/spucosun/10.21]' infer_lr='[1e-2,1e-3,1e-4]' infer_weight_decay='[1e-5, 1e-4, 1e-3, 1e-2]' infer_num_iters='[1000, 45000]' lr='[1e-3,1e-4,1e-5]' weight_decay='[1e-5, 1e-4, 1e-3, 1e-2]' num_epochs='[40, 100]' arch=cliprn50 only_train_projection=True pretrained=True -m 16 --stage-trials
```
