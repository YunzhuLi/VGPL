# Visual Grounding of Learned Physical Models

Yunzhu Li, Toru Lin*, Kexin Yi*, Daniel M. Bear, Daniel L. K. Yamins, Jiajun Wu, Joshua B. Tenenbaum, and Antonio Torralba

**ICML 2020**
[[website]](http://visual-physics-grounding.csail.mit.edu/) [[paper]](https://arxiv.org/abs/2004.13664) [[video]](https://www.youtube.com/watch?v=P_LrG0lzc-0&feature=youtu.be)


## Evaluate the trained model on the validation data

Download the validation data from the following links. Unzip and put them in the `data/` folder.
- MassRope [[DropBox]](https://www.dropbox.com/s/l3fx5onv21ti72p/data_MassRope_valid.zip?dl=0) (2.8 GB)
- RigidFall [[DropBox]](https://www.dropbox.com/s/o3ehs4s4p13kuy6/data_RigidFall_valid.zip?dl=0) (7.35 GB)

### 1. Dynamics Prior

Type the following commands to evaluate the dynamics prior. You can also take a look at [[VGPL-Dynamics-Prior]](https://github.com/YunzhuLi/VGPL-Dynamics-Prior), which we prepared as a stand-alone module for dynamics prediction.

    bash scripts/dynamics/eval_MassRope_dy.sh
    bash scripts/dynamics/eval_RigidFall_dy.sh
    
You will be able to generate videos like the following

![](imgs/MassRope.gif)  ![](imgs/RigidFall.gif)
    
### 2. Parameter Estimation

Type the following command to evaluate the model's performance on parameter estimation on 50 testing examples.

    bash scripts/parameter/eval_MassRope_param.sh
    bash scripts/parameter/eval_RigidFall_param.sh
    
- MassRope: Average error ratio: 2.8812% (std: 1.2736)
- RigidFall: Average error ratio: 3.7455% (std: 2.6649)
    
### 3. Position Refinement and Rigidness estimation

Type the following command to evaluate the model's performance on position refinement and rigidness estimation on 50 testing examples.

    bash scripts/position/eval_MassRope_pos.sh
    bash scripts/position/eval_RigidFall_pos.sh
    
Position Mean Squared Error (scaled by 1e4)
- MassRope: Before refinement 1.9584, After refinement 0.4782
- RigidFall: Before refinement 1.9700, After refinement 1.4500

### 4. Forward prediction using the inference results

Type the following command to evaluate the model's performance on forward prediction using the inference results.

    bash scripts/forward/eval_MassRope_fwd.sh
    bash scripts/forward/eval_RigidFall_fwd.sh
