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


if args.stage == 'forward':
    # load dynamics module and position refinement module
    if args.eval_epoch_pos < 0:
        model_name = 'net_best.pth'
    else:
        model_name = 'net_epoch_%d_iter_%d.pth' % (args.eval_epoch_pos, args.eval_iter_pos)

    model_path = os.path.join(args.outf.replace('forward', 'pos'), model_name)
    print("Loading network from %s" % model_path)

    pretrained_dict = torch.load(model_path)
    model_dict = model.state_dict()

    pretrained_dict = {
        k: v for k, v in pretrained_dict.items() \
        if ('dynamics_predictor' in k or 'position_and_group_refiner' in k) and k in model_dict}
    model.load_state_dict(pretrained_dict, strict=False)


    # load parameter estimation module
    if args.eval_epoch_param < 0:
        model_name = 'net_best.pth'
    else:
        model_name = 'net_epoch_%d_iter_%d.pth' % (args.eval_epoch_param, args.eval_iter_param)

    model_path = os.path.join(args.outf.replace('forward', 'param'), model_name)
    print("Loading network from %s" % model_path)

    pretrained_dict = torch.load(model_path)
    model_dict = model.state_dict()

    pretrained_dict = {
        k: v for k, v in pretrained_dict.items() \
        if ('param_estimator' in k) and k in model_dict}
    model.load_state_dict(pretrained_dict, strict=False)


else:
    AssertionError("Unsupported stage %s, using other evaluation scripts" % args.stage)

model.eval()

if use_gpu:
    model = model.cuda()


# define loss
particle_dist_loss = torch.nn.MSELoss()


infos = np.arange(10)
phase = 'valid'
st_idx = 30
ed_idx = st_idx + args.sequence_length
observ_length = 10


loss_refined_fwd = []

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

    ''' inference '''
    with torch.set_grad_enabled(False):
        # refine the position and grouping
        inputs = [
            attrs,
            particles_percept[:, :observ_length],
            groups_percept[:, :observ_length],
            Rrs[:, :observ_length],
            Rss[:, :observ_length],
            n_particle, n_shape, args.n_instance]

        # particles_refined: B x observ_length x (n_p + n_s) x state_dim
        # p_rigid: B x observ_length x n_instance
        # p_instance: B x observ_length x n_p x n_instance
        particles_refined, p_rigid, p_instance = model.refine_position_and_grouping(inputs)
        p_rigid_np = p_rigid.data.cpu().numpy()
        p_rigid_np[p_rigid_np >= 0.5] = 1
        p_rigid_np[p_rigid_np < 0.5] = 0
        p_rigid = torch.tensor(p_rigid_np).cuda()

        # physics_param: B x observ_length x n_p
        physics_param = model.estimate_param(inputs)[:, None, :].repeat(1, observ_length, 1)



    # memory: B x mem_nlayer x (n_particle + n_shape) x nf_memory
    # for now, only used as a placeholder
    memory_init = model.init_memory(B, n_particle + n_shape)


    '''
    future prediction using with all inferred results
    '''

    loss_refined_fwd_cur = []
    particles_refined_fwd = []

    with torch.set_grad_enabled(False):
        # if env is RigidFall, all particles are rigid particles, use ground truth p_rigid
        if args.env == 'RigidFall':
            groups_refined = groups_gt[0], p_instance, physics_param
        else:
            groups_refined = p_rigid, p_instance, physics_param

        ''' loss on particle positions after dynamics prediction '''
        # calculate graph
        loss_nxt = 0
        loss_counter = 0
        n_forward_prediction_step = args.sequence_length - observ_length

        for batch_idx in range(B):
            # state_cur: 1 x n_his x (n_p + n_s) x state_dim
            state_cur = particles_refined[batch_idx:batch_idx + 1, observ_length - n_his:observ_length]
            state_cur = state_cur.view(1, n_his, n_particle + n_shape, args.state_dim)

            for step_idx in range(observ_length, seq_length):
                # Rr_cur, Rs_cur: n_rel x (n_p + n_s)
                _, _, Rr_cur, Rs_cur = prepare_input(state_cur[0, -1], n_particle, n_shape, args, var=True)
                Rr_cur = torch.FloatTensor(Rr_cur).cuda()
                Rs_cur = torch.FloatTensor(Rs_cur).cuda()

                if use_gpu:
                    Rr_cur = Rr_cur.cuda()
                    Rs_cur = Rs_cur.cuda()

                group_cur = [d[batch_idx:batch_idx + 1, 0] for d in groups_refined]

                inputs = [attrs[0:1], state_cur, Rr_cur[None, ...], Rs_cur[None, ...], memory_init[0:1], group_cur]

                # predict the next step using the dynamics model
                # pred_pos (unnormalized): 1 x n_p x state_dim
                # pred_motion_norm (normalized): 1 x n_p x state_dim
                pred_pos, pred_motion_norm = model.predict_dynamics(inputs)

                # append the predicted particle position to state_cur
                # pred_pose: 1 x (n_p + n_s) x state_dim
                # state_cur: 1 x n_his x (n_p + n_s) x state_dim
                pred_pos = torch.cat([pred_pos, particles_gt[batch_idx:batch_idx + 1, step_idx, n_particle:]], 1)
                state_cur = torch.cat([state_cur[:, 1:], pred_pos.unsqueeze(1)], 1)

                # state_pred: 1 x n_p x state_dim
                # state_gt: 1 x n_p x state_dim
                state_pred = state_cur[:, -1, :n_particle]
                state_gt = particles_gt[batch_idx:batch_idx + 1, step_idx, :n_particle]

                loss_refined_fwd_cur.append(F.mse_loss(state_pred, state_gt).item())
                particles_refined_fwd.append(state_pred[0].data.cpu().numpy())

    loss_refined_fwd.append(loss_refined_fwd_cur)


    '''
    render results
    '''

    if args.vispy:

        ### render in VisPy
        import vispy.scene
        from vispy import app
        from vispy.visuals import transforms

        particle_size = 0.01
        border = 0.025
        height = 1.3
        y_rotate_deg = -45.0


        def y_rotate(obj, deg=y_rotate_deg):
            tr = vispy.visuals.transforms.MatrixTransform()
            tr.rotate(deg, (0, 1, 0))
            obj.transform = tr

        def add_floor(v):
            # add floor
            floor_length = 3.0
            w, h, d = floor_length, floor_length, border
            b1 = vispy.scene.visuals.Box(width=w, height=h, depth=d, color=[0.8, 0.8, 0.8, 1], edge_color='black')
            y_rotate(b1)
            v.add(b1)

            # adjust position of box
            mesh_b1 = b1.mesh.mesh_data
            v1 = mesh_b1.get_vertices()
            c1 = np.array([0., -particle_size - border, 0.], dtype=np.float32)
            mesh_b1.set_vertices(np.add(v1, c1))

            mesh_border_b1 = b1.border.mesh_data
            vv1 = mesh_border_b1.get_vertices()
            cc1 = np.array([0., -particle_size - border, 0.], dtype=np.float32)
            mesh_border_b1.set_vertices(np.add(vv1, cc1))

        def update_box_states(boxes, last_states, curr_states):
            v = curr_states[0] - last_states[0]
            if args.verbose_data:
                print("box states:", last_states, curr_states)
                print("box velocity:", v)

            tr = vispy.visuals.transforms.MatrixTransform()
            tr.rotate(y_rotate_deg, (0, 1, 0))

            for i, box in enumerate(boxes):
                # use v to update box translation
                trans = (curr_states[i][0], curr_states[i][1], curr_states[i][2])
                box.transform = tr * vispy.visuals.transforms.STTransform(translate=trans)

        def translate_box(b, x, y, z):
            mesh_b = b.mesh.mesh_data
            v = mesh_b.get_vertices()
            c = np.array([x, y, z], dtype=np.float32)
            mesh_b.set_vertices(np.add(v, c))

            mesh_border_b = b.border.mesh_data
            vv = mesh_border_b.get_vertices()
            cc = np.array([x, y, z], dtype=np.float32)
            mesh_border_b.set_vertices(np.add(vv, cc))

        def add_box(v, w=0.1, h=0.1, d=0.1, x=0.0, y=0.0, z=0.0):
            """
            Add a box object to the scene view
            :param v: view to which the box should be added
            :param w: width
            :param h: height
            :param d: depth
            :param x: x center
            :param y: y center
            :param z: z center
            :return: None
            """
            # render background box
            b = vispy.scene.visuals.Box(width=w, height=h, depth=d, color=[0.8, 0.8, 0.8, 1], edge_color='black')
            y_rotate(b)
            v.add(b)

            # adjust position of box
            translate_box(b, x, y, z)

            return b

        def calc_box_init(x, z):
            boxes = []

            # floor
            boxes.append([x, z, border, 0., -particle_size / 2, 0.])

            # left wall
            boxes.append([border, z, (height + border), -particle_size / 2, 0., 0.])

            # right wall
            boxes.append([border, z, (height + border), particle_size / 2, 0., 0.])

            # back wall
            boxes.append([(x + border * 2), border, (height + border)])

            # front wall (disabled when colored)
            # boxes.append([(x + border * 2), border, (height + border)])

            return boxes

        def add_container(v, box_x, box_z):
            boxes = calc_box_init(box_x, box_z)
            visuals = []
            for b in boxes:
                if len(b) == 3:
                    visual = add_box(v, b[0], b[1], b[2])
                elif len(b) == 6:
                    visual = add_box(v, b[0], b[1], b[2], b[3], b[4], b[5])
                else:
                    raise AssertionError("Input should be either length 3 or length 6")
                visuals.append(visual)
            return visuals


        c = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
        view = c.central_widget.add_view()

        if args.env == 'RigidFall':
            view.camera = vispy.scene.cameras.TurntableCamera(fov=50, azimuth=45, elevation=20, distance=2, up='+y')
            # set instance colors
            instance_colors = create_instance_colors(args.n_instance)

            # render floor
            add_floor(view)

        if args.env == 'MassRope':
            view.camera = vispy.scene.cameras.TurntableCamera(fov=30, azimuth=0, elevation=20, distance=8, up='+y')

            # set instance colors
            n_string_particles = 15
            instance_colors = create_instance_colors(args.n_instance)

            # render floor
            add_floor(view)


        # render particles
        p1 = vispy.scene.visuals.Markers()
        p1.antialias = 0  # remove white edge

        y_rotate(p1)

        view.add(p1)

        # set animation
        t_step = 0


        '''
        set up data for rendering
        '''
        #0 - particles_gt_fwd
        #1 - particles_refined_fwd
        #2 - images_fwd

        # particles: (seq_length - observ_length) x n_p x 3
        # images: (seq_length - observ_length) x H x W x 3

        particles_gt_fwd = particles_gt[0, observ_length:].data.cpu().numpy()
        particles_refined_fwd = np.stack(particles_refined_fwd)
        images_fwd = data_vision[1][observ_length:]

        print('particles_gt_fwd', particles_gt_fwd.shape)
        print('particles_refined_fwd', particles_refined_fwd.shape)
        print('images_fwd', images_fwd.shape)

        vis_length = seq_length - observ_length

        # create directory to save images if not exist
        vispy_dir = args.evalf + "/vispy"
        os.system('mkdir -p ' + vispy_dir)

        ### store the visual images
        for t_actual in range(vis_length):
            img = cv2.resize(images_fwd[t_actual], (800, 600), interpolation=cv2.INTER_CUBIC)
            img_path = os.path.join(vispy_dir, "vision_{}_{}.png".format(str(idx_episode), str(t_actual)))
            cv2.imwrite(img_path, img)


        # groups_percept: seq_length x n_p x n_group
        if args.env in ['RigidFall']:
            groups_percept = [None, groups_percept.data.cpu().numpy()[0]]
        elif args.env in ['MassRope']:
            groups_percept = [None, groups_percept.data.cpu().numpy()[0]]

        # p_rigid: n_instance
        # p_instance: n_p x n_instance
        # physics_param: n_p
        # groups_percept: seq_length x n_p x n_group
        groups_refined = [d.data.cpu().numpy()[0, ...] for d in groups_refined]
        groups_gt = [d.data.cpu().numpy()[0, ...] for d in groups_gt]


        def update(event):
            global p1
            global t_step
            global colors

            if t_step < vis_length:
                if t_step == 0:
                    print("Rendering ground truth")

                t_actual = t_step

                colors = convert_groups_to_colors(
                    [None if d is None else d[0] for d in groups_gt],
                    n_particle, args.n_instance,
                    instance_colors=instance_colors, env=args.env)

                colors = np.clip(colors, 0., 1.)

                p1.set_data(particles_gt_fwd[t_actual, :n_particle],
                            edge_color='black', face_color=colors)

                # render for ground truth
                img = c.render()
                img_path = os.path.join(vispy_dir, "gt_{}_{}.png".format(str(idx_episode), str(t_actual)))
                vispy.io.write_png(img_path, img)


            elif vis_length <= t_step < vis_length * 2:
                if t_step == vis_length:
                    print("Rendering refinement result")

                t_actual = t_step - vis_length

                colors = convert_groups_to_colors(
                    [None if d is None else d[0] for d in groups_refined],
                    n_particle, args.n_instance,
                    instance_colors=instance_colors, env=args.env)

                colors = np.clip(colors, 0., 1.)

                p1.set_data(particles_refined_fwd[t_actual, :n_particle],
                            edge_color='black', face_color=colors)

                # render for refinement
                img = c.render()
                img_path = os.path.join(vispy_dir, "refined_{}_{}.png".format(str(idx_episode), str(t_actual)))
                vispy.io.write_png(img_path, img)


            else:
                # discarded frames
                pass

            # time forward
            t_step += 1


        # start animation
        timer = app.Timer()
        timer.connect(update)
        timer.start(interval=1. / 60., iterations=vis_length * 2)

        c.show()
        app.run()

        # render video for evaluating grouping result
        if args.stage in ['forward']:
            print("Render video for evaluating grouping result")

            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(
                os.path.join(args.evalf, 'vid_%d_%d_vispy.avi' % (args.discard_frames, idx_episode)),
                fourcc, 3, (800 * 3, 600))

            for step in range(vis_length):
                vision_path = os.path.join(args.evalf, 'vispy', 'vision_%d_%d.png' % (idx_episode, step))
                refined_path = os.path.join(args.evalf, 'vispy', 'refined_%d_%d.png' % (idx_episode, step))
                gt_path = os.path.join(args.evalf, 'vispy', 'gt_%d_%d.png' % (idx_episode, step))

                vision = cv2.imread(vision_path)
                refined = cv2.imread(refined_path)
                gt = cv2.imread(gt_path)

                frame = np.zeros((600, 800 * 3, 3), dtype=np.uint8)
                frame[:, :800] = vision
                frame[:, 800:1600] = refined
                frame[:, 1600:2400] = gt

                out.write(frame)

            out.release()



''' store the results'''

scale = 1e4

loss_refined_fwd = np.array(loss_refined_fwd) * scale
loss_refined_fwd = np.transpose(loss_refined_fwd, (1, 0))
rec_path = os.path.join(args.evalf, 'loss_refined_fwd.npy')
print('Save results to %s' % rec_path)
np.save(rec_path, loss_refined_fwd)


''' print the results '''

print('loss_refined_fwd', loss_refined_fwd.shape)

print()
for i in range(seq_length - observ_length):
    print(i, ', %.4f (%.4f)' % (np.mean(loss_refined_fwd[i]), np.std(loss_refined_fwd[i])))



