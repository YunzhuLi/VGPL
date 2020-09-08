import argparse
import copy
import os
import time
import cv2
import matplotlib.pyplot as plt

import numpy as np
import scipy.misc
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from config import gen_args
from data import load_data, get_scene_info, normalize_scene_param
from data import get_env_group, prepare_input, denormalize
from models import Model, ChamferLoss
from utils import add_log, convert_groups_to_colors
from utils import create_instance_colors, set_seed, Tee, count_parameters

import matplotlib.pyplot as plt


args = gen_args()
set_seed(args.random_seed)

os.system('mkdir -p ' + args.evalf)
tee = Tee(os.path.join(args.evalf, 'eval.log'), 'w')


### evaluating

data_names = args.data_names

use_gpu = torch.cuda.is_available()

# create model and load weights
model = Model(args, use_gpu)
print("model_kp #params: %d" % count_parameters(model))

if args.eval_epoch < 0:
    model_name = 'net_best.pth'
else:
    model_name = 'net_epoch_%d_iter_%d.pth' % (args.eval_epoch, args.eval_iter)

model_path = os.path.join(args.outf, model_name)
print("Loading network from %s" % model_path)

if args.stage == 'pos':
    pretrained_dict = torch.load(model_path)
    model_dict = model.state_dict()

    # only load parameters in dynamics_predictor
    pretrained_dict = {
        k: v for k, v in pretrained_dict.items() \
        if ('dynamics_predictor' in k or 'position_and_group_refiner' in k) and k in model_dict}
    model.load_state_dict(pretrained_dict, strict=False)

else:
    AssertionError("Unsupported stage %s, using other evaluation scripts" % args.stage)

model.eval()

if use_gpu:
    model = model.cuda()


# define loss
particle_dist_loss = torch.nn.MSELoss()


infos = np.arange(50)
phase = 'valid'
st_idx = 30
ed_idx = st_idx + args.sequence_length


rigidness_rec = []
pos_refinement_rec = []
pos_perception_rec = []

for idx_episode in range(len(infos)):

    print("Rollout %d / %d" % (idx_episode, len(infos)))


    '''
    load data (refer to data.py)
    '''
    # load images
    data_path = os.path.join(args.dataf, '%s_vision' % phase, '%d.h5' % infos[idx_episode])
    data_vision = load_data(['positions', 'images'], data_path)
    data_vision = [d[st_idx:ed_idx] for d in data_vision]
    # print('positions', data_vision[0].shape)
    # print('images', data_vision[1].shape)

    # load perception results
    loss_dir = 'l2'
    data_path = os.path.join(
        args.dataf, 'perception', loss_dir, phase, str(infos[idx_episode]),
        str(st_idx - args.discard_frames) + '.h5')
    data_perception = load_data(['positions', 'groups'], data_path)

    # load ground truth
    attrs, particles_percept, groups_percept, particles_gt, Rrs, Rss = [], [], [], [], [], []
    max_n_rel = 0
    for t in range(st_idx, ed_idx):
        # load data
        data_path = os.path.join(args.dataf, phase, str(infos[idx_episode]), str(t) + '.h5')
        data = load_data(args.data_names, data_path)

        # load scene param
        if t == st_idx:
            n_particle, n_shape, scene_params = get_scene_info(data)

        # concate the position of the detected particles and the positions of the shapes
        # position_particle: n_p x state_dim
        # position_shape: n_s x state_dim
        # position_perception: (n_p + n_s) x state_dim
        # group_perception: n_p x n_group
        position_particle, position_shape = data[0][:n_particle], data[0][n_particle:]
        position_perception = np.concatenate([data_perception[0][t - st_idx], position_shape], 0)
        group_perception = data_perception[1][t - st_idx]

        attr, particle, Rr, Rs = prepare_input(position_perception, n_particle, n_shape, args)

        max_n_rel = max(max_n_rel, Rr.size(0))

        attrs.append(attr)
        particles_percept.append(particle.numpy())
        particles_gt.append(data[0])
        groups_percept.append(group_perception)
        Rrs.append(Rr)
        Rss.append(Rs)

    # groups_gt
    groups_gt = get_env_group(args, n_particle, torch.FloatTensor(scene_params)[None, ...], use_gpu=use_gpu)

    # attr: (n_p + n_s) x attr_dim
    # particles_percept (unnormalized): seq_length x (n_p + n_s) x state_dim
    # scene_params: param_dim
    attr = torch.FloatTensor(attrs[0])
    particles_percept = torch.FloatTensor(np.stack(particles_percept))
    scene_params = torch.FloatTensor(scene_params)

    # pad the relation set
    # Rr, Rs: seq_length x n_rel x (n_p + n_s)
    for i in range(len(Rrs)):
        Rr, Rs = Rrs[i], Rss[i]
        Rr = torch.cat([Rr, torch.zeros(max_n_rel - Rr.size(0), n_particle + n_shape)], 0)
        Rs = torch.cat([Rs, torch.zeros(max_n_rel - Rs.size(0), n_particle + n_shape)], 0)
        Rrs[i], Rss[i] = Rr, Rs
    Rr = torch.FloatTensor(np.stack(Rrs))
    Rs = torch.FloatTensor(np.stack(Rss))

    # particles_gt (unnormalized): seq_length x (n_p + n_s) x state_dim
    # groups_percept: seq_length x n_p x n_group
    #   RigidFall: n_group == 3
    #   MassRope: n_group == 2
    particles_gt = torch.FloatTensor(particles_gt)
    groups_percept = F.softmax(torch.FloatTensor(groups_percept), -1)


    '''
    add batch dimension to data & move to gpu
    '''
    # attrs: B x (n_p + n_s) x attr_dim
    # particles_percept: B x seq_length x (n_p + n_s) x state_dim
    # particles_gt: B x seq_length x (n_p + n_s) x state_dim
    # groups_percept: B x seq_length x n_p x n_group
    # scene_params: B x param_dim
    # Rrs, Rss: B x seq_length x n_rel x (n_p + n_s)
    attrs = attr[None, ...]
    particles_percept = particles_percept[None, ...]
    particles_gt = particles_gt[None, ...]
    groups_percept = groups_percept[None, ...]
    scene_params = scene_params[None, ...]
    Rrs = Rr[None, ...]
    Rss = Rs[None, ...]

    if use_gpu:
        attrs = attrs.cuda()
        particles_percept = particles_percept.cuda()
        particles_gt = particles_gt.cuda()
        groups_percept = groups_percept.cuda()
        Rrs, Rss = Rrs.cuda(), Rss.cuda()


    '''
    statistics info
    '''
    B = 1
    seq_length = args.sequence_length
    n_his = args.n_his


    '''
    model prediction
    '''
    # p_rigid: B x n_instance
    # p_instance: B x n_particle x n_instance
    # physics_param: B x n_particle
    p_rigid_gt, p_instance_gt, physics_param_gt = get_env_group(args, n_particle, scene_params, use_gpu=use_gpu)

    # repeat the ground truth group information along the time axis
    # p_rigid: B x seq_length x n_instance
    # p_instance: B x seq_length x n_particle x n_instance
    # physics_param: B x seq_length x n_particle
    groups_gt = [p_rigid_gt[:, None, :].repeat(1, seq_length, 1),
                 p_instance_gt[:, None, :, :].repeat(1, seq_length, 1, 1),
                 physics_param_gt[:, None, :].repeat(1, seq_length, 1)]

    ''' record rigidness overtime '''
    rigidness_rec_cur = []
    with torch.set_grad_enabled(False):
        for observ_length in range(args.sequence_length):
            # refine the position and grouping
            inputs = [
                attrs,
                particles_percept[:, :observ_length + 1],
                groups_percept[:, :observ_length + 1],
                Rrs[:, :observ_length + 1],
                Rss[:, :observ_length + 1],
                n_particle, n_shape, args.n_instance]
            # particles_refined: B x seq_length x (n_p + n_s) x state_dim
            # p_rigid: B x seq_length x n_instance
            # p_instance: B x seq_length x n_p x n_instance
            particles_refined, p_rigid, p_instance = model.refine_position_and_grouping(inputs)

            ''' see how many particles are rigid '''
            p_rigid_np = p_rigid.data.cpu().numpy()
            p_instance_np = p_instance.data.cpu().numpy()
            n_instance = p_instance_np.shape[3]

            pr = []
            for idx_instance in range(n_instance):
                '''
                n = np.sum(p_instance_np[:, :, :, idx_instance] > 0.5, 2)
                print(idx_instance,
                    np.mean(p_rigid_np[..., idx_instance]), np.std(p_rigid_np[..., idx_instance]),
                    np.mean(n), np.std(n))
                '''
                pr.append(np.mean(p_rigid_np[..., idx_instance]))

            rigidness_rec_cur.append(pr)


            ''' record position refinment '''
            if observ_length + 1 == 10:
                pos_refinement_rec.append(F.mse_loss(
                    particles_refined[:, :, :n_particle],
                    particles_gt[:, :observ_length + 1, :n_particle]).item())
                pos_perception_rec.append(F.mse_loss(
                    particles_percept[:, :observ_length + 1, :n_particle],
                    particles_gt[:, :observ_length + 1, :n_particle]).item())


    rigidness_rec.append(rigidness_rec_cur)



''' store the results'''

rigidness_rec = np.array(rigidness_rec)
rigidness_rec = np.transpose(rigidness_rec, (2, 1, 0))

rec_path = os.path.join(args.evalf, 'rigidness.npy')
print('Save results to %s' % rec_path)
np.save(rec_path, rigidness_rec)


pos_refinement_rec = np.array(pos_refinement_rec)
pos_perception_rec = np.array(pos_perception_rec)

rec_path = os.path.join(args.evalf, 'pos_refinement.npy')
print('Save results to %s' % rec_path)
np.save(rec_path, pos_refinement_rec)

rec_path = os.path.join(args.evalf, 'pos_perception.npy')
print('Save results to %s' % rec_path)
np.save(rec_path, pos_perception_rec)


print()
print("Position MSE")
scale = 1e4
print("Before refinement %.4f" % np.mean(pos_perception_rec * scale))
print("After refinement %.4f" % np.mean(pos_refinement_rec * scale))



''' plot for MassRope '''

def plot_data_median(ax, data, color, label):
    m, lo, hi = np.median(data, 1), np.quantile(data, 0.25, 1), np.quantile(data, 0.75, 1)
    T = len(m)
    x = np.arange(1, T + 1)
    ax.plot(x, m, '-', color=color, alpha=0.8, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.2)


if args.env == 'MassRope':
    instance_names = ['Mass', 'Rope']
    rigidness_rec[1] = 1. - rigidness_rec[1]

    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5), dpi=200)
    colors = ['r', 'b', 'g']

    plt.axvline(x=10, color='orange', linestyle='--', linewidth=3)

    for i in range(rigidness_rec.shape[0]):
        plot_data_median(ax, rigidness_rec[i], color=colors[i], label=instance_names[i])

    plt.grid(True)
    plt.legend(loc='best', fontsize=12)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Mean probability', fontsize=12)
    plt.xticks([1, 5, 10, 15, 20])
    plt.ylim([0.93, 1.006])
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(args.evalf, 'rigidness_%s.pdf' % args.env))
    plt.savefig(os.path.join(args.evalf, 'rigidness_%s.png' % args.env))
    plt.show()

