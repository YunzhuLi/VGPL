import argparse
import copy
import os
import time
import cv2

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

if args.stage == 'param':
    model.load_state_dict(torch.load(model_path))
else:
    AssertionError("Unsupported stage %s, using other evaluation scripts" % args.stage)

model.eval()

if use_gpu:
    model = model.cuda()



infos = np.arange(50)
phase = 'valid'
st_idx = args.discard_frames
ed_idx = st_idx + args.sequence_length

loss_param_rec = []

for idx_episode in range(len(infos)):

    print("Rollout %d / %d" % (idx_episode, len(infos)))

    '''
    load data (refer to data.py)
    '''
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
    groups_gt = get_env_group(args, n_particle, scene_params, use_gpu=use_gpu)
    p_rigid_gt, p_instance_gt, physics_param_gt = groups_gt

    ''' plot param estimation overtime '''
    with torch.set_grad_enabled(False):
        observ_length = 10
        # refine the position and grouping
        inputs = [
            attrs,
            particles_percept[:, :observ_length],
            groups_percept[:, :observ_length],
            Rrs[:, :observ_length],
            Rss[:, :observ_length],
            n_particle, n_shape, args.n_instance]

        # physics_param: B x n_p
        physics_param = model.estimate_param(inputs)

        param_gt = torch.mean(physics_param_gt).item()
        param_pred = torch.mean(physics_param).item()

        error_ratio = F.l1_loss(physics_param, physics_param_gt).item() / 2.
        print('GT param: %.4f, Predicted param: %.4f, Ratio: %.4f' % (
            param_gt, param_pred, error_ratio))

        loss_param_rec.append(error_ratio)


print()
print("Average error ratio: %.4f%% (%.4f)" % (
    np.mean(loss_param_rec) * 100., np.std(loss_param_rec) * 100.))
