import os
import time
import sys
import copy

import multiprocessing as mp
from progressbar import ProgressBar

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from config import gen_args
from data import PhysicsFleXDataset
from data import prepare_input, get_scene_info, get_env_group
from models import Model, ChamferLoss
from utils import make_graph, check_gradient, set_seed, AverageMeter, get_lr, Tee
from utils import count_parameters, my_collate


args = gen_args()
set_seed(args.random_seed)

os.system('mkdir -p ' + args.dataf)
os.system('mkdir -p ' + args.outf)

tee = Tee(os.path.join(args.outf, 'train.log'), 'w')


### training

# load training data

phases = ['train', 'valid'] if args.eval == 0 else ['valid']
datasets = {phase: PhysicsFleXDataset(args, phase) for phase in phases}

for phase in phases:
    if args.gen_data:
        datasets[phase].gen_data(args.env)
    else:
        datasets[phase].load_data(args.env)

dataloaders = {phase: DataLoader(
    datasets[phase],
    batch_size=args.batch_size,
    shuffle=True if phase == 'train' else False,
    num_workers=args.num_workers,
    collate_fn=my_collate) for phase in phases}

# create model and train
use_gpu = torch.cuda.is_available()
model = Model(args, use_gpu)

print("model #params: %d" % count_parameters(model))


# checkpoint to reload model from
model_path = None

# resume training of a saved model (if given)
if args.resume == 0:
    if args.stage == 'dy':
        print("Randomly initialize the model's parameters")

    elif args.stage in ['pos', 'param']:
        if args.stage == 'pos':
            outf_dy = args.outf.replace('files_pos', 'files_dy')
        elif args.stage == 'param':
            outf_dy = args.outf.replace('files_param', 'files_dy')
        model_path = os.path.join(outf_dy, 'net_epoch_%d_iter_%d.pth' % (
            args.resume_epoch, args.resume_iter))
        print("Loading saved ckp from %s" % model_path)
        pretrained_dict = torch.load(model_path)
        model_dict = model.state_dict()

        # only load parameters in dynamics_predictor
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() \
            if 'dynamics_predictor' in k and k in model_dict}
        model.load_state_dict(pretrained_dict, strict=False)

elif args.resume == 1:
    model_path = os.path.join(args.outf, 'net_epoch_%d_iter_%d.pth' % (
        args.resume_epoch, args.resume_iter))
    print("Loading saved ckp from %s" % model_path)

    if args.stage == 'dy':
        pretrained_dict = torch.load(model_path)
        model_dict = model.state_dict()

        # only load parameters in dynamics_predictor
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() \
            if 'dynamics_predictor' in k and k in model_dict}
        model.load_state_dict(pretrained_dict, strict=False)

    elif args.stage == 'pos':
        pretrained_dict = torch.load(model_path)
        model_dict = model.state_dict()

        # only load parameters in dynamics_predictor
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() \
            if ('dynamics_predictor' in k or 'position_and_group_refiner' in k) and k in model_dict}
        model.load_state_dict(pretrained_dict, strict=False)

    else:
        model.load_state_dict(torch.load(model_path))

# optimizer
if args.stage == 'dy':
    params = model.dynamics_predictor.parameters()
elif args.stage == 'pos':
    params = model.position_and_group_refiner.parameters()
elif args.stage == 'param':
    params = model.param_estimator.parameters()
else:
    raise AssertionError("unknown stage: %s" % args.stage)

if args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(
        params, lr=args.lr, betas=(args.beta1, 0.999))
elif args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=0.9)
else:
    raise AssertionError("unknown optimizer: %s" % args.optimizer)

# reduce learning rate when a metric has stopped improving
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=3, verbose=True)

# define loss
particle_dist_loss = torch.nn.L1Loss()

if use_gpu:
    model = model.cuda()

# log args
print(args)

# start training
st_epoch = args.resume_epoch if args.resume_epoch > 0 else 0
best_valid_loss = np.inf

for epoch in range(st_epoch, args.n_epoch):

    for phase in phases:

        model.train(phase == 'train')

        meter_loss = AverageMeter()
        meter_loss_raw = AverageMeter()

        meter_loss_ref = AverageMeter()
        meter_loss_nxt = AverageMeter()

        meter_loss_param = AverageMeter()


        bar = ProgressBar(max_value=len(dataloaders[phase]))

        for i, data in bar(enumerate(dataloaders[phase])):
            # each "data" is a trajectory of sequence_length time steps

            if args.stage == 'dy':
                # attrs: B x (n_p + n_s) x attr_dim
                # particles: B x seq_length x (n_p + n_s) x state_dim
                # n_particles: B
                # n_shapes: B
                # scene_params: B x param_dim
                # Rrs, Rss: B x seq_length x n_rel x (n_p + n_s)
                attrs, particles, n_particles, n_shapes, scene_params, Rrs, Rss = data

                if use_gpu:
                    attrs = attrs.cuda()
                    particles = particles.cuda()
                    Rrs, Rss = Rrs.cuda(), Rss.cuda()

                # statistics
                B = attrs.size(0)
                n_particle = n_particles[0].item()
                n_shape = n_shapes[0].item()

                # p_rigid: B x n_instance
                # p_instance: B x n_particle x n_instance
                # physics_param: B x n_particle
                groups_gt = get_env_group(args, n_particle, scene_params, use_gpu=use_gpu)

                # memory: B x mem_nlayer x (n_particle + n_shape) x nf_memory
                # for now, only used as a placeholder
                memory_init = model.init_memory(B, n_particle + n_shape)

                with torch.set_grad_enabled(phase == 'train'):
                    # state_cur (unnormalized): B x n_his x (n_p + n_s) x state_dim
                    state_cur = particles[:, :args.n_his]

                    # Rrs_cur, Rss_cur: B x n_rel x (n_p + n_s)
                    Rr_cur = Rrs[:, args.n_his - 1]
                    Rs_cur = Rss[:, args.n_his - 1]

                    # predict the velocity at the next time step
                    inputs = [attrs, state_cur, Rr_cur, Rs_cur, memory_init, groups_gt]

                    # pred_pos (unnormalized): B x n_p x state_dim
                    # pred_motion_norm (normalized): B x n_p x state_dim
                    pred_pos, pred_motion_norm = model.predict_dynamics(inputs)

                    # concatenate the state of the shapes
                    # pred_pos (unnormalized): B x (n_p + n_s) x state_dim
                    gt_pos = particles[:, args.n_his]
                    pred_pos = torch.cat([pred_pos, gt_pos[:, n_particle:]], 1)

                    # gt_motion_norm (normalized): B x (n_p + n_s) x state_dim
                    # pred_motion_norm (normalized): B x (n_p + n_s) x state_dim
                    gt_motion = particles[:, args.n_his] - particles[:, args.n_his - 1]
                    mean_d, std_d = model.stat[2:]
                    gt_motion_norm = (gt_motion - mean_d) / std_d
                    pred_motion_norm = torch.cat([pred_motion_norm, gt_motion_norm[:, n_particle:]], 1)

                    loss = F.l1_loss(pred_motion_norm[:, :n_particle], gt_motion_norm[:, :n_particle])
                    loss_raw = F.l1_loss(pred_pos, gt_pos)

                    meter_loss.update(loss.item(), B)
                    meter_loss_raw.update(loss_raw.item(), B)

                if i % args.log_per_iter == 0:
                    print()
                    print('%s epoch[%d/%d] iter[%d/%d] LR: %.6f, loss: %.6f (%.6f), loss_raw: %.8f (%.8f)' % (
                        phase, epoch, args.n_epoch, i, len(dataloaders[phase]), get_lr(optimizer),
                        loss.item(), meter_loss.avg, loss_raw.item(), meter_loss_raw.avg))


            elif args.stage in ['pos']:
                # attrs: B x (n_p + n_s) x attr_dim
                # particles_percept: B x seq_length x (n_p + n_s) x state_dim
                # particles_gt: B x seq_length x (n_p + n_s) x state_dim
                # groups_percept: B x seq_length x n_p x n_group
                # n_particles: B
                # n_shapes: B
                # scene_params: B x param_dim
                # Rrs, Rss: B x seq_length x n_rel x (n_p + n_s)
                attrs, particles_percept, particles_gt, groups_percept, n_particles, n_shapes, scene_params, Rrs, Rss = data

                '''
                print('attrs', attrs.size())
                print('particles_percept', particles_percept.size())
                print('particles_gt', particles_gt.size())
                print('groups_percept', groups_percept.size())
                print('n_particles', n_particles.size())
                print('n_shapes', n_shapes.size())
                print('scene_params', scene_params.size())
                print('Rrs', Rrs.size())
                print('Rss', Rss.size())
                '''

                if use_gpu:
                    attrs = attrs.cuda()
                    particles_percept = particles_percept.cuda()
                    particles_gt = particles_gt.cuda()
                    groups_percept = groups_percept.cuda()
                    Rrs, Rss = Rrs.cuda(), Rss.cuda()

                # statistics
                B = attrs.size(0)
                n_particle = n_particles[0].item()
                n_shape = n_shapes[0].item()
                seq_length = args.sequence_length
                n_his = args.n_his

                # p_rigid: B x n_instance
                # p_instance: B x n_particle x n_instance
                # physics_param: B x n_particle
                p_rigid, p_instance, physics_param = get_env_group(args, n_particle, scene_params, use_gpu=use_gpu)

                # repeat the ground truth group information along the time axis
                # p_rigid: B x seq_length x n_instance
                # p_instance: B x seq_length x n_particle x n_instance
                # physics_param: B x seq_length x n_particle
                groups_gt = [p_rigid[:, None, :].repeat(1, seq_length, 1),
                             p_instance[:, None, :, :].repeat(1, seq_length, 1, 1),
                             physics_param[:, None, :].repeat(1, seq_length, 1)]

                # memory: B x mem_nlayer x (n_particle + n_shape) x nf_memory
                # for now, only used as a placeholder
                memory_init = model.init_memory(B, n_particle + n_shape)

                with torch.set_grad_enabled(phase == 'train'):
                    # refine the position and grouping
                    inputs = [attrs, particles_percept, groups_percept, Rrs, Rss, n_particle, n_shape, args.n_instance]
                    # particles_refined: B x seq_length x (n_p + n_s) x state_dim
                    # p_rigid: B x seq_length x n_instance
                    # p_instance: B x seq_length x n_p x n_instance
                    particles_refined, p_rigid, p_instance = model.refine_position_and_grouping(inputs)

                    # if env is RigidFall, all particles are rigid particles, use ground truth p_rigid
                    if args.env == 'RigidFall':
                        groups_refined = groups_gt[0], p_instance, groups_gt[2]
                    else:
                        groups_refined = p_rigid, p_instance, groups_gt[2]


                    ''' chamfer loss on refined particle positions '''
                    loss_raw, loss_ref = 0., 0.

                    for step_idx in range(seq_length - 1):
                        # state_raw: B x n_p x (2 * state_dim)
                        # state_ref: B x n_p x (2 * state_dim)
                        # state_gt: B x n_p x (2 * state_dim)
                        state_raw = particles_percept[:, step_idx:step_idx + 2, :n_particle]
                        state_raw = state_raw.transpose(1, 2).contiguous().view(B, n_particle, 2 * args.state_dim)
                        state_ref = particles_refined[:, step_idx:step_idx + 2, :n_particle]
                        state_ref = state_ref.transpose(1, 2).contiguous().view(B, n_particle, 2 * args.state_dim)
                        state_gt = particles_gt[:, step_idx:step_idx + 2, :n_particle]
                        state_gt = state_gt.transpose(1, 2).contiguous().view(B, n_particle, 2 * args.state_dim)

                        loss_raw += particle_dist_loss(state_raw, state_gt) / (seq_length - 1)
                        loss_ref += particle_dist_loss(state_ref, state_gt) / (seq_length - 1)


                    ''' chamfer loss on particle positions after dynamics prediction '''
                    # calculate graph
                    loss_nxt = 0
                    loss_counter = 0
                    if args.env in ['MassRope', 'RigidFall']:
                        n_forward_prediction_step = 1

                    for batch_idx in range(B):
                        for step_idx in range(n_his, seq_length - n_forward_prediction_step + 1):
                            # state_cur: 1 x n_his x (n_p + n_s) x state_dim
                            state_cur = particles_refined[batch_idx:batch_idx + 1, step_idx - n_his:step_idx]
                            state_cur = state_cur.view(1, n_his, n_particle + n_shape, args.state_dim)

                            for forward_idx in range(n_forward_prediction_step):
                                idx_cur = step_idx + forward_idx

                                # Rr_cur, Rs_cur: n_rel x (n_p + n_s)
                                _, _, Rr_cur, Rs_cur = prepare_input(state_cur[0, -1], n_particle, n_shape, args, var=True)
                                Rr_cur = torch.FloatTensor(Rr_cur).cuda()
                                Rs_cur = torch.FloatTensor(Rs_cur).cuda()

                                if use_gpu:
                                    Rr_cur = Rr_cur.cuda()
                                    Rs_cur = Rs_cur.cuda()

                                group_cur = [d[batch_idx:batch_idx + 1, idx_cur - 1] for d in groups_refined]

                                inputs = [attrs[0:1], state_cur, Rr_cur[None, ...], Rs_cur[None, ...],
                                          memory_init[0:1], group_cur]

                                # predict the next step using the dynamics model
                                # pred_pos (unnormalized): 1 x n_p x state_dim
                                # pred_motion_norm (normalized): 1 x n_p x state_dim
                                pred_pos, pred_motion_norm = model.predict_dynamics(inputs)

                                # append the predicted particle position to state_cur
                                # pred_pose: 1 x (n_p + n_s) x state_dim
                                # state_cur: 1 x n_his x (n_p + n_s) x state_dim
                                pred_pos = torch.cat([pred_pos, particles_gt[batch_idx:batch_idx + 1, idx_cur, n_particle:]], 1)
                                state_cur = torch.cat([state_cur[:, 1:], pred_pos.unsqueeze(1)], 1)

                                # state_pred: 1 x n_p x (2 * state_dim)
                                # state_gt: 1 x n_p x (2 * state_dim)
                                state_pred = state_cur[:, -2:, :n_particle]
                                state_pred = state_pred.transpose(1, 2).contiguous().view(1, n_particle, 2 * args.state_dim)
                                state_gt = particles_gt[batch_idx:batch_idx + 1, idx_cur - 1:idx_cur + 1, :n_particle]
                                state_gt = state_gt.transpose(1, 2).contiguous().view(1, n_particle, 2 * args.state_dim)

                                loss_nxt += particle_dist_loss(state_pred, state_gt)
                                loss_counter += 1

                    loss_nxt = loss_nxt / loss_counter

                    loss = loss_ref + loss_nxt

                    meter_loss.update(loss.item(), B)
                    meter_loss_ref.update(loss_ref.item(), B)
                    meter_loss_nxt.update(loss_nxt.item(), B)
                    meter_loss_raw.update(loss_raw.item(), B)


                if i % args.log_per_iter == 0:
                    print()
                    print('%s epoch[%d/%d] iter[%d/%d] LR: %.6f, loss: %.6f (%.6f), loss_ref: %.6f (%.6f), loss_nxt: %.6f (%.6f), loss_raw: %.6f (%.6f)' % (
                        phase, epoch, args.n_epoch, i, len(dataloaders[phase]), get_lr(optimizer),
                        loss.item(), meter_loss.avg,
                        loss_ref.item(), meter_loss_ref.avg,
                        loss_nxt.item(), meter_loss_nxt.avg,
                        loss_raw.item(), meter_loss_raw.avg))


            elif args.stage in ['param']:
                # attrs: B x (n_p + n_s) x attr_dim
                # particles_percept: B x seq_length x (n_p + n_s) x state_dim
                # particles_gt: B x seq_length x (n_p + n_s) x state_dim
                # groups_percept: B x seq_length x n_p x n_group
                # n_particles: B
                # n_shapes: B
                # scene_params: B x param_dim
                # Rrs, Rss: B x seq_length x n_rel x (n_p + n_s)
                attrs, particles_percept, particles_gt, groups_percept, n_particles, n_shapes, scene_params, Rrs, Rss = data

                '''
                print('attrs', attrs.size())
                print('particles_percept', particles_percept.size())
                print('particles_gt', particles_gt.size())
                print('groups_percept', groups_percept.size())
                print('n_particles', n_particles.size())
                print('n_shapes', n_shapes.size())
                print('scene_params', scene_params.size())
                print('Rrs', Rrs.size())
                print('Rss', Rss.size())
                '''

                if use_gpu:
                    attrs = attrs.cuda()
                    particles_percept = particles_percept.cuda()
                    particles_gt = particles_gt.cuda()
                    groups_percept = groups_percept.cuda()
                    Rrs, Rss = Rrs.cuda(), Rss.cuda()

                # statistics
                B = attrs.size(0)
                n_particle = n_particles[0].item()
                n_shape = n_shapes[0].item()
                seq_length = args.sequence_length
                n_his = args.n_his

                # p_rigid: B x n_instance
                # p_instance: B x n_particle x n_instance
                # physics_param: B x n_particle
                groups_gt = get_env_group(args, n_particle, scene_params, use_gpu=use_gpu)
                p_rigid_gt, p_instance_gt, physics_param_gt = groups_gt

                # memory: B x mem_nlayer x (n_particle + n_shape) x nf_memory
                # for now, only used as a placeholder
                memory_init = model.init_memory(B, n_particle + n_shape)

                with torch.set_grad_enabled(phase == 'train'):
                    # refine the position and grouping
                    inputs = [attrs, particles_percept, groups_percept, Rrs, Rss, n_particle, n_shape, args.n_instance]
                    # physics_param: B x n_p
                    physics_param = model.estimate_param(inputs)

                    group_refined = p_rigid_gt, p_instance_gt, physics_param

                    loss_param = F.l1_loss(physics_param, physics_param_gt)
                    meter_loss_param.update(loss_param.item(), B)

                    loss = 0.
                    loss_counter = 0
                    n_forward_prediction_step = 2

                    for batch_idx in range(B):
                        for step_idx in range(n_his, n_his + 1):
                            # state_cur: 1 x n_his x (n_p + n_s) x state_dim
                            state_cur = particles_gt[batch_idx:batch_idx + 1, step_idx - n_his:step_idx]

                            for forward_idx in range(n_forward_prediction_step):
                                idx_cur = step_idx + forward_idx

                                # Rr_cur, Rs_cur: n_rel x (n_p + n_s)
                                _, _, Rr_cur, Rs_cur = prepare_input(state_cur[0, -1], n_particle, n_shape, args, var=True)
                                Rr_cur = torch.FloatTensor(Rr_cur).cuda()
                                Rs_cur = torch.FloatTensor(Rs_cur).cuda()

                                if use_gpu:
                                    Rr_cur = Rr_cur.cuda()
                                    Rs_cur = Rs_cur.cuda()

                                group_cur = [d[batch_idx:batch_idx + 1] for d in group_refined]

                                inputs = [attrs[0:1], state_cur, Rr_cur[None, ...], Rs_cur[None, ...],
                                          memory_init[0:1], group_cur]

                                # predict the next step using the dynamics model
                                # pred_pos (unnormalized): 1 x n_p x state_dim
                                # pred_motion_norm (normalized): 1 x n_p x state_dim
                                pred_pos, pred_motion_norm = model.predict_dynamics(inputs)

                                gt_motion = particles_gt[batch_idx:batch_idx + 1, idx_cur] - \
                                        particles_gt[batch_idx:batch_idx + 1, idx_cur - 1]
                                mean_d, std_d = model.stat[2:]
                                gt_motion_norm = (gt_motion[:, :n_particle] - mean_d) / std_d

                                loss += F.l1_loss(pred_motion_norm, gt_motion_norm)
                                loss_counter += 1

                                # append the predicted particle position to state_cur
                                # pred_pose: 1 x (n_p + n_s) x state_dim
                                # state_cur: 1 x n_his x (n_p + n_s) x state_dim
                                pred_pos = torch.cat([pred_pos, particles_gt[batch_idx:batch_idx + 1, idx_cur, n_particle:]], 1)
                                state_cur = torch.cat([state_cur[:, 1:], pred_pos.unsqueeze(1)], 1)

                    loss = loss / loss_counter

                    meter_loss.update(loss.item(), B)


                if i % args.log_per_iter == 0:
                    print()
                    print('%s epoch[%d/%d] iter[%d/%d] LR: %.6f, loss: %.6f (%.6f), loss_param: %.6f (%.6f)' % (
                        phase, epoch, args.n_epoch, i, len(dataloaders[phase]), get_lr(optimizer),
                        loss.item(), meter_loss.avg, loss_param.item(), meter_loss_param.avg))



            # update model parameters
            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if phase == 'train' and i > 0 and i % args.ckp_per_iter == 0:
                model_path = '%s/net_epoch_%d_iter_%d.pth' % (args.outf, epoch, i)
                torch.save(model.state_dict(), model_path)



        print('%s epoch[%d/%d] Loss: %.6f, Best valid: %.6f' % (
            phase, epoch, args.n_epoch, meter_loss.avg, best_valid_loss))

        if phase == 'valid' and not args.eval:
            scheduler.step(meter_loss.avg)
            if meter_loss.avg < best_valid_loss:
                best_valid_loss = meter_loss.avg
                torch.save(model.state_dict(), '%s/net_best.pth' % (args.outf))
