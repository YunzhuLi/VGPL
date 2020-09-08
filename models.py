import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from data import prepare_input


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        s = x.size()
        x = self.model(x.view(-1, s[-1]))
        return x.view(list(s[:-1]) + [-1])


class Propagator(nn.Module):
    def __init__(self, input_size, output_size, residual=False):
        super(Propagator, self).__init__()

        self.residual = residual

        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, res=None):
        s_x = x.size()

        if self.residual:
            s_res = res.size()

        x = self.linear(x.view(-1, s_x[-1]))

        if self.residual:
            x += res.view(-1, s_res[-1])

        x = self.relu(x).view(list(s_x[:-1]) + [-1])
        return x


class ParticlePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticlePredictor, self).__init__()

        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        s_x = x.size()

        x = x.view(-1, s_x[-1])
        x = self.relu(self.linear_0(x))
        x = self.relu(self.linear_1(x))

        return self.linear_2(x).view(list(s_x[:-1]) + [-1])


class DynamicsPredictor(nn.Module):
    def __init__(self, args, residual=False, use_gpu=False):

        super(DynamicsPredictor, self).__init__()

        self.args = args

        n_his = args.n_his
        attr_dim = args.attr_dim
        state_dim = args.state_dim
        mem_dim = args.nf_effect * args.mem_nlayer

        nf_particle = args.nf_particle
        nf_relation = args.nf_relation
        nf_effect = args.nf_effect

        self.nf_effect = nf_effect

        self.use_gpu = use_gpu
        self.residual = residual

        self.quat_offset = torch.FloatTensor([1., 0., 0., 0.])
        if use_gpu:
            self.quat_offset = self.quat_offset.cuda()

        # ParticleEncoder
        input_dim = attr_dim + 1 + n_his * state_dim * 2 + mem_dim
        self.particle_encoder = Encoder(input_dim, nf_particle, nf_effect)

        # RelationEncoder
        self.relation_encoder = Encoder(input_dim * 2 + 1, nf_relation, nf_effect)

        # ParticlePropagator
        self.particle_propagator = Propagator(nf_effect * 2, nf_effect, self.residual)

        # RelationPropagator
        self.relation_propagator = Propagator(nf_effect * 3, nf_effect)

        # ParticlePredictor
        self.rigid_predictor = ParticlePredictor(nf_effect, nf_effect, 7)
        self.non_rigid_predictor = ParticlePredictor(nf_effect, nf_effect, state_dim)

    def rotation_matrix_from_quaternion(self, params):
        # params: (B * n_instance) x 4
        # w, x, y, z

        one = torch.ones(1, 1)
        zero = torch.zeros(1, 1)
        if self.use_gpu:
            one = one.cuda()
            zero = zero.cuda()

        # multiply the rotation matrix from the right-hand side
        # the matrix should be the transpose of the conventional one

        # Reference
        # http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm

        params = params / torch.norm(params, dim=1, keepdim=True)
        w, x, y, z = \
                params[:, 0].view(-1, 1, 1), params[:, 1].view(-1, 1, 1), \
                params[:, 2].view(-1, 1, 1), params[:, 3].view(-1, 1, 1)

        rot = torch.cat((
            torch.cat((one - y * y * 2 - z * z * 2, x * y * 2 + z * w * 2, x * z * 2 - y * w * 2), 2),
            torch.cat((x * y * 2 - z * w * 2, one - x * x * 2 - z * z * 2, y * z * 2 + x * w * 2), 2),
            torch.cat((x * z * 2 + y * w * 2, y * z * 2 - x * w * 2, one - x * x * 2 - y * y * 2), 2)), 1)

        # rot: (B * n_instance) x 3 x 3
        return rot

    def forward(self, inputs, stat, verbose=0):
        args = self.args
        verbose = args.verbose_model
        mean_p, std_p, mean_d, std_d = stat

        # attrs: B x N x attr_dim
        # state (unnormalized): B x n_his x N x state_dim
        # Rr_cur, Rs_cur: B x n_rel x N
        # memory: B x mem_nlayer x N x nf_memory
        # group:
        #   p_rigid: B x n_instance
        #   p_instance: B x n_particle x n_instance
        #   physics_param: B x n_particle
        attrs, state, Rr_cur, Rs_cur, memory, group = inputs
        p_rigid, p_instance, physics_param = group

        # Rr_cur_t, Rs_cur_t: B x N x n_rel
        Rr_cur_t = Rr_cur.transpose(1, 2).contiguous()
        Rs_cur_t = Rs_cur.transpose(1, 2).contiguous()

        # number of particles that need prediction
        B, N = attrs.size(0), attrs.size(1)
        n_p = p_instance.size(1)
        n_s = attrs.size(1) - n_p

        n_his = args.n_his
        state_dim = args.state_dim

        # state_norm (normalized): B x n_his x N x state_dim
        # [0, n_his - 1): state_residual
        # [n_his - 1, n_his): the current position
        state_res_norm = (state[:, 1:] - state[:, :-1] - mean_d) / std_d
        state_cur_norm = (state[:, -1:] - mean_p) / std_p
        state_norm = torch.cat([state_res_norm, state_cur_norm], 1)

        # state_norm_t (normalized): B x N x (n_his * state_dim)
        state_norm_t = state_norm.transpose(1, 2).contiguous().view(B, N, n_his * state_dim)

        # add offset to center-of-mass for rigids to attr
        # offset: B x N x (n_his * state_dim)
        offset = torch.zeros(B, N, n_his * state_dim)
        if self.use_gpu:
            offset = offset.cuda()

        # p_rigid_per_particle: B x n_p x 1
        p_rigid_per_particle = torch.sum(p_instance * p_rigid[:, None, :], 2, keepdim=True)

        # instance_center: B x n_instance x (n_his * state_dim)
        instance_center = p_instance.transpose(1, 2).bmm(state_norm_t[:, :n_p])
        instance_center /= torch.sum(p_instance, 1).unsqueeze(-1) + args.eps

        # c_per_particle: B x n_p x (n_his * state_dim)
        # particle offset: B x n_p x (n_his * state_dim)
        c_per_particle = p_instance.bmm(instance_center)
        c = (1 - p_rigid_per_particle) * state_norm_t[:, :n_p] + p_rigid_per_particle * c_per_particle
        offset[:, :n_p] = state_norm_t[:, :n_p] - c

        # memory_t: B x N x (mem_nlayer * nf_memory)
        # physics_param: B x N x 1
        # attrs: B x N x (attr_dim + 1 + n_his * state_dim + mem_nlayer * nf_memory)
        memory_t = memory.transpose(1, 2).contiguous().view(B, N, -1)
        physics_param_s = torch.zeros(B, n_s, 1)
        if self.use_gpu:
            physics_param_s = physics_param_s.cuda()
        physics_param = torch.cat([physics_param[:, :, None], physics_param_s], 1)
        attrs = torch.cat([attrs, physics_param, offset, memory_t], 2)

        # group info
        # g: B x N x n_instance
        g = p_instance
        g_s = torch.zeros(B, n_s, args.n_instance)
        if self.use_gpu:
            g_s = g_s.cuda()
        g = torch.cat([g, g_s], 1)

        # receiver_attr, sender_attr
        # attrs_r: B x n_rel x -1
        # attrs_s: B x n_rel x -1
        attrs_r = Rr_cur.bmm(attrs)
        attrs_s = Rs_cur.bmm(attrs)

        # receiver_state, sender_state
        # state_norm_r: B x n_rel x -1
        # state_norm_s: B x n_rel x -1
        state_norm_r = Rr_cur.bmm(state_norm_t)
        state_norm_s = Rs_cur.bmm(state_norm_t)

        # receiver_group, sender_group
        # group_r: B x n_rel x -1
        # group_s: B x n_rel x -1
        group_r = Rr_cur.bmm(g)
        group_s = Rs_cur.bmm(g)
        group_diff = torch.sum(torch.abs(group_r - group_s), 2, keepdim=True)

        # particle encode
        if verbose:
            print('attrs_r', attrs_r.shape, 'state_norm_r', state_norm_r.shape)
        particle_encode = self.particle_encoder(torch.cat([attrs, state_norm_t], 2))
        particle_effect = particle_encode
        if verbose:
            print("particle encode:", particle_encode.size())

        # calculate relation encoding
        relation_encode = self.relation_encoder(
            torch.cat([attrs_r, attrs_s, state_norm_r, state_norm_s, group_diff], 2))
        if verbose:
            print("relation encode:", relation_encode.size())

        for i in range(args.pstep):
            if verbose:
                print("pstep", i)

            # effect_r, effect_s: B x n_rel x nf
            effect_r = Rr_cur.bmm(particle_effect)
            effect_s = Rs_cur.bmm(particle_effect)

            # calculate relation effect
            # effect_rel: B x n_rel x nf
            effect_rel = self.relation_propagator(
                torch.cat([relation_encode, effect_r, effect_s], 2))
            if verbose:
                print("relation effect:", effect_rel.size())

            # calculate particle effect by aggregating relation effect
            # effect_rel_agg: B x N x nf
            effect_rel_agg = Rr_cur_t.bmm(effect_rel)

            # calculate particle effect
            # particle_effect: B x N x nf
            particle_effect = self.particle_propagator(
                torch.cat([particle_encode, effect_rel_agg], 2),
                res=particle_effect)
            if verbose:
                 print("particle effect:", particle_effect.size())

        # non_rigid_motion: B x n_p x state_dim
        non_rigid_motion = self.non_rigid_predictor(particle_effect[:, :n_p].contiguous())

        # rigid motion
        # instance effect: B x n_instance x nf_effect
        n_instance = p_instance.size(2)
        instance_effect = p_instance.transpose(1, 2).bmm(particle_effect[:, :n_p])

        # rigid motion
        # instance_rigid_params: (B * n_instance) x 7
        instance_rigid_params = self.rigid_predictor(instance_effect).view(B * n_instance, 7)

        # R: (B * n_instance) x 3 x 3
        R = self.rotation_matrix_from_quaternion(instance_rigid_params[:, :4] + self.quat_offset)
        if verbose:
            print("Rotation matrix", R.size(), "should be (B x n_instance, 3, 3)")

        b = instance_rigid_params[:, 4:] * std_d + mean_d
        b = b.view(B * n_instance, 1, state_dim)
        if verbose:
            print("b", b.size(), "should be (B x n_instance, 1, state_dim)")

        p_0 = state[:, -1:, :n_p]
        p_0 = p_0.repeat(1, n_instance, 1, 1).view(B * n_instance, n_p, state_dim)
        if verbose:
            print("p_0", p_0.size(), "should be (B x n_instance, n_p, state_dim)")

        c = instance_center[:, :, -3:] * std_p + mean_p
        c = c.view(B * n_instance, 1, state_dim)
        if verbose:
            print("c", c.size(), "should be (B x n_instance, 1, state_dim)")

        p_1 = torch.bmm(p_0 - c, R) + b + c
        if verbose:
            print("p_1", p_1.size(), "should be (B x n_instance, n_p, state_dim)")

        # rigid_motion: B x n_instance x n_p x state_dim
        rigid_motion = (p_1 - p_0).view(B, n_instance, n_p, state_dim)
        rigid_motion = (rigid_motion - mean_d) / std_d

        # merge rigid and non-rigid motion
        # rigid_motion      (B x n_instance x n_p x state_dim)
        # non_rigid_motion  (B x n_p x state_dim)
        pred_motion = (1. - p_rigid_per_particle) * non_rigid_motion
        pred_motion += torch.sum(
            p_rigid[:, :, None, None] * \
            p_instance.transpose(1, 2)[:, :, :, None] * \
            rigid_motion, 1)

        pred_pos = state[:, -1, :n_p] + (pred_motion * std_d + mean_d)

        if verbose:
            print('pred_pos', pred_pos.size())

        # pred_pos (unnormalized): B x n_p x state_dim
        # pred_motion_norm (normalized): B x n_p x state_dim
        return pred_pos, pred_motion



class PropNet(nn.Module):
    def __init__(self, nf_input, nf_hidden, nf_output):
        super(PropNet, self).__init__()

        self.node_encoder = Encoder(nf_input, nf_hidden, nf_hidden)
        self.edge_encoder = Encoder(nf_input * 2, nf_hidden, nf_hidden)

        self.node_propagator = Propagator(nf_hidden * 2, nf_hidden, residual=True)
        self.edge_propagator = Propagator(nf_hidden * 3, nf_hidden)

        self.node_predictor = ParticlePredictor(nf_hidden * 2, nf_hidden, nf_output)

    def forward(self, x, Rr, Rs, pstep=2):
        # x: B x N x nf_input
        # Rr, Rs: B x n_rel x N

        # Rr_t, Rs_t: B x N x n_rel
        Rr_t = Rr.transpose(1, 2).contiguous()
        Rs_t = Rs.transpose(1, 2).contiguous()

        # node_encode: B x N x nf_hidden
        node_encode = self.node_encoder(x)
        node_effect = node_encode

        # edge_encode: B x n_rel x nf_hidden
        x_r = Rr.bmm(x)
        x_s = Rs.bmm(x)
        edge_encode = self.edge_encoder(torch.cat([x_r, x_s], -1))

        for step in range(pstep):
            # node_effect_r, node_effect_s: B x n_rel x nf_hidden
            node_effect_r = Rr.bmm(node_effect)
            node_effect_s = Rs.bmm(node_effect)

            # edge_effect: B x n_rel x nf_hidden
            edge_effect = self.edge_propagator(
                torch.cat([edge_encode, node_effect_r, node_effect_s], 2))

            # node_effect_agg: B x N x nf_hidden
            node_effect_agg = Rr_t.bmm(edge_effect)

            # node_effect: B x N x nf_hidden
            node_effect = self.node_propagator(
                torch.cat([node_encode, node_effect_agg], 2),
                res=node_effect)

        # node_reps: B x N x nf_hidden
        node_reps = self.node_predictor(torch.cat([node_encode, node_effect], 2))

        return node_reps



class PositionAndGroupRefiner(nn.Module):
    def __init__(self, args):
        super(PositionAndGroupRefiner, self).__init__()

        self.args = args

        # PropNet for spatial message passing
        nf_input = args.attr_dim + args.state_dim + args.n_group_percept
        nf_hidden = args.nf_pos
        nf_output = args.nf_effect
        self.propnet = PropNet(nf_input, nf_hidden, nf_output)

        # Bidirectional GRU for temporal message passing
        self.bi_gru = nn.GRU(input_size=args.nf_effect,
                          hidden_size=nf_hidden,
                          num_layers=2,
                          batch_first=False,
                          bidirectional=True)

        # position refiner
        self.position_refiner = nn.Sequential(
            nn.Linear(2 * args.nf_effect, args.nf_effect),
            nn.ReLU(),
            nn.Linear(args.nf_effect, args.state_dim))

        # p_rigid predictor
        self.p_rigid_predictor = nn.Sequential(
            nn.Linear(2 * args.nf_effect, args.nf_effect),
            nn.ReLU(),
            nn.Linear(args.nf_effect, args.n_instance),
            nn.Sigmoid())

        # p_instance predictor
        self.p_instance_predictor = nn.Sequential(
            nn.Linear(2 * args.nf_effect, args.nf_effect),
            nn.ReLU(),
            nn.Linear(args.nf_effect, args.n_instance))

    def forward(self, inputs, stat, use_gpu=True):
        args = self.args
        mean_p, std_p, mean_d, std_d = stat

        # attr: B x (n_p + n_s) x attr_dim
        # state: B x seq_length x (n_p + n_s) x state_dim
        # group_percept: B x seq_length x n_p x n_instance
        # Rr, Rs: B x seq_length x n_rel x (n_p + n_s)
        attr, state, group_percept, Rr, Rs, n_p, n_s, n_instance = inputs

        state_norm = (state - mean_p) / std_p

        B, seq_length, _, n_group = group_percept.size()
        n_rel = Rr.size(2)

        ''' PropNet for spatial message passing '''
        pad = torch.zeros(B, seq_length, n_s, n_group)
        if use_gpu:
            pad = pad.cuda()

        # inputs: (B * seq_length) x (n_p + n_s) x (attr_dim + state_dim + n_group)
        inputs = torch.cat([
            attr[:, None, :, :].repeat(1, seq_length, 1, 1),
            state_norm,
            torch.cat([group_percept, pad], 2)], 3)
        inputs = inputs.view(B * seq_length, n_p + n_s, args.attr_dim + args.state_dim + n_group)

        # reps: B x seq_length x n_p x nf_effect
        reps = self.propnet(
            inputs,
            Rr.view(B * seq_length, n_rel, n_p + n_s),
            Rs.view(B * seq_length, n_rel, n_p + n_s),
            pstep=2)
        reps = reps.view(B, seq_length, n_p + n_s, args.nf_effect)[:, :, :n_p]

        ''' Bidirectional LSTM for temporal message passing '''
        # bi_output: seq_length x (B * n_p) x (num_directions * nf_effect)
        # bi_hidden: (num_layers * num_directions) x (B * n_p) x nf_effect
        bi_output, bi_hidden = self.bi_gru(
            reps.transpose(0, 1).contiguous().view(seq_length, B * n_p, args.nf_effect))

        ''' Predict particles_refined '''
        # state_residual_norm: B x seq_length x n_p x state_dim
        state_residual_norm = self.position_refiner(
            bi_output.view(seq_length * B * n_p, 2 * args.nf_effect))
        state_residual_norm = state_residual_norm.view(
            seq_length, B, n_p, args.state_dim).transpose(0, 1)

        # state_refined: B x seq_length x (n_p + n_s) x state_dim
        state_refined_norm = state_norm[:, :, :n_p] + state_residual_norm
        state_refined = state_refined_norm * std_p + mean_p
        state_refined = torch.cat([state_refined, state[:, :, n_p:]], 2)

        ''' Calculate p_rigid and p_instance '''
        '''
        # reps: B x n_p x (nf_effect * 2)
        reps = torch.cat([bi_output[-1, :, :args.nf_effect], bi_output[0, :, args.nf_effect:]], 1)
        reps = reps.view(B, n_p, args.nf_effect * 2)

        p_rigid = self.p_rigid_predictor(torch.mean(reps, 1)).view(B, n_instance)
        p_instance = self.p_instance_predictor(reps).view(B, n_p, n_instance)
        p_instance = F.softmax(p_instance + torch.mean(group_percept, 1), -1)
        '''
        # reps: B x seq_length x n_p x (nf_effect * 2)
        reps = bi_output.view(seq_length, B, n_p, 2 * args.nf_effect).transpose(0, 1)
        p_rigid = self.p_rigid_predictor(torch.mean(reps, 2)).view(B, seq_length, n_instance)
        p_instance = group_percept

        ''' Return '''
        # state_refined: B x seq_length x (n_p + n_s) x state_dim
        # p_rigid: B x seq_length x n_instance
        # p_instance: B x seq_length x n_p x n_instance
        return state_refined, p_rigid, p_instance



class ParameterEstimator(nn.Module):
    def __init__(self, args):
        super(ParameterEstimator, self).__init__()

        self.args = args

        # PropNet for spatial message passing
        nf_input = args.attr_dim + args.state_dim + args.n_group_percept
        nf_hidden = args.nf_pos
        nf_output = args.nf_effect
        self.propnet = PropNet(nf_input, nf_hidden, nf_output)

        # Bidirectional GRU for temporal message passing
        self.bi_gru = nn.GRU(input_size=args.nf_effect,
                          hidden_size=nf_hidden,
                          num_layers=2,
                          batch_first=False,
                          bidirectional=True)

        # param estimator
        self.param_estimator = nn.Sequential(
            nn.Linear(2 * args.nf_effect, args.nf_effect),
            nn.ReLU(),
            nn.Linear(args.nf_effect, 1),
            nn.Tanh())

    def forward(self, inputs, stat, use_gpu=True):
        args = self.args
        mean_p, std_p, mean_d, std_d = stat

        # attr: B x (n_p + n_s) x attr_dim
        # state: B x seq_length x (n_p + n_s) x state_dim
        # group_percept: B x seq_length x n_p x n_instance
        # Rr, Rs: B x seq_length x n_rel x (n_p + n_s)
        attr, state, group_percept, Rr, Rs, n_p, n_s, n_instance = inputs

        state_norm = (state - mean_p) / std_p

        B, seq_length, _, n_group = group_percept.size()
        n_rel = Rr.size(2)

        ''' PropNet for spatial message passing '''
        pad = torch.zeros(B, seq_length, n_s, n_group)
        if use_gpu:
            pad = pad.cuda()

        # inputs: (B * seq_length) x (n_p + n_s) x (attr_dim + state_dim + n_group)
        inputs = torch.cat([
            attr[:, None, :, :].repeat(1, seq_length, 1, 1),
            state_norm,
            torch.cat([group_percept, pad], 2)], 3)
        inputs = inputs.view(B * seq_length, n_p + n_s, args.attr_dim + args.state_dim + n_group)

        # reps: B x seq_length x n_p x nf_effect
        reps = self.propnet(
            inputs,
            Rr.view(B * seq_length, n_rel, n_p + n_s),
            Rs.view(B * seq_length, n_rel, n_p + n_s),
            pstep=2)
        reps = reps.view(B, seq_length, n_p + n_s, args.nf_effect)[:, :, :n_p]

        ''' Bidirectional LSTM for temporal message passing '''
        # bi_output: seq_length x (B * n_p) x (num_directions * nf_effect)
        # bi_hidden: (num_layers * num_directions) x (B * n_p) x nf_effect
        bi_output, bi_hidden = self.bi_gru(
            reps.transpose(0, 1).contiguous().view(seq_length, B * n_p, args.nf_effect))

        ''' Calculate p_rigid and p_instance '''
        # reps: B x n_p x (nf_effect * 2)
        reps = torch.cat([bi_output[-1, :, :args.nf_effect], bi_output[0, :, args.nf_effect:]], 1)
        reps = reps.view(B, n_p, args.nf_effect * 2)

        physics_param = self.param_estimator(torch.mean(reps, 1)).view(B, 1).repeat(1, n_p)

        ''' Return '''
        # physics_param: B x n_p
        return physics_param




class Model(nn.Module):
    def __init__(self, args, use_gpu):

        super(Model, self).__init__()

        self.args = args
        self.use_gpu = use_gpu

        self.dt = torch.FloatTensor([args.dt])
        mean_p = torch.FloatTensor(args.mean_p)
        std_p = torch.FloatTensor(args.std_p)
        mean_d = torch.FloatTensor(args.mean_d)
        std_d = torch.FloatTensor(args.std_d)

        if use_gpu:
            self.dt = self.dt.cuda()
            mean_p = mean_p.cuda()
            std_p = std_p.cuda()
            mean_d = mean_d.cuda()
            std_d = std_d.cuda()

        self.stat = [mean_p, std_p, mean_d, std_d]

        # PropNet to predict forward dynamics
        self.dynamics_predictor = DynamicsPredictor(args, use_gpu=use_gpu)

        # Position and group refiner
        self.position_and_group_refiner = PositionAndGroupRefiner(args)

        # Parameter estimator
        self.param_estimator = ParameterEstimator(args)

    def init_memory(self, B, N):
        """
        memory  (B, mem_layer, N, nf_memory)
        """
        mem = torch.zeros(B, self.args.mem_nlayer, N, self.args.nf_effect)
        if self.use_gpu:
            mem = mem.cuda()
        return mem

    def predict_dynamics(self, inputs):
        """
        return:
        ret - predicted position of all particles, shape (n_particles, 3)
        """
        ret = self.dynamics_predictor(inputs, self.stat, self.args.verbose_model)
        return ret

    def refine_position_and_grouping(self, inputs):
        """
        return:
        particles_refined: B x seq_length x (n_p + n_s) x state_dim
        p_rigid: B x n_instance
        p_instance: B x n_p x n_instance
        """
        ret = self.position_and_group_refiner(inputs, self.stat)
        return ret

    def estimate_param(self, inputs):
        """
        return:
        physics_param: B x n_p
        """
        ret = self.param_estimator(inputs, self.stat)
        return ret



class ChamferLoss(torch.nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def chamfer_distance(self, x, y):
        # x: [B, N, D]
        # y: [B, M, D]
        x = x[:, :, None, :].repeat(1, 1, y.size(1), 1) # x: [B, N, M, D]
        y = y[:, None, :, :].repeat(1, x.size(1), 1, 1) # y: [B, N, M, D]
        dis = torch.norm(torch.add(x, -y), 2, dim=3)    # dis: [B, N, M]
        dis_xy = torch.mean(torch.min(dis, dim=2)[0])   # dis_xy: mean over N
        dis_yx = torch.mean(torch.min(dis, dim=1)[0])   # dis_yx: mean over M

        return dis_xy + dis_yx

    def __call__(self, pred, label):
        # pred: [B, N, D]
        # label: [B, M, D]
        return self.chamfer_distance(pred, label)
