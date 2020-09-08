# Visual Grounding of Learned Physical Models

Yunzhu Li, Toru Lin*, Kexin Yi*, Daniel M. Bear, Daniel L. K. Yamins, Jiajun Wu, Joshua B. Tenenbaum, and Antonio Torralba

**ICML 2020**
[[website]](http://visual-physics-grounding.csail.mit.edu/) [[paper]](https://arxiv.org/abs/2004.13664) [[video]](https://www.youtube.com/watch?v=P_LrG0lzc-0&feature=youtu.be)


## Evaluate the trained model on the validation data

Link to validation data
- MassRope [[DropBox]](https://www.dropbox.com/s/l3fx5onv21ti72p/data_MassRope_valid.zip?dl=0) (2.8 GB)
- RigidFall [[DropBox]](https://www.dropbox.com/s/o3ehs4s4p13kuy6/data_RigidFall_valid.zip?dl=0) (7.35 GB)

Dynamics

    bash scripts/dynamics/eval_MassRope_dy.sh
    bash scripts/dynamics/eval_RigidFall_dy.sh
    
Parameter Estimation

    bash scripts/parameter/eval_MassRope_param.sh
    bash scripts/parameter/eval_RigidFall_param.sh
    
Position Refinement and Rigidness estimation

    bash scripts/position/eval_MassRope_pos.sh
    bash scripts/position/eval_RigidFall_pos.sh

Forward prediction after dynamics-guided inference

    bash scripts/forward/eval_MassRope_fwd.sh
    bash scripts/forward/eval_RigidFall_fwd.sh
