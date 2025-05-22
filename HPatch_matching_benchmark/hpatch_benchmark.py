import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from scipy.io import loadmat
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

methods = ['ORB', 'ORB+Boost-B-FP32', 'ORB+Boost-B-FP16', 'ORB+Boost-B-INT8']
names = ['ORB', 'ORB+Boost-B-FP32', 'ORB+Boost-B-FP16', 'ORB+Boost-B-INT8']
colors = ['orange', 'green', 'blue', 'red']
linestyles = ['-', '--', '-', '-']

n_i = 52
n_v = 56

dataset_path = '/media/wei/T7/hpatches-sequences-release'

lim = [1, 15]
rng = np.arange(lim[0], lim[1] + 1)

def mnn_matcher(descriptors_a, descriptors_b):
    sim = descriptors_a @ descriptors_b.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=descriptors_a.device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t().cpu().numpy()

def benchmark_features(read_feats):
    seq_names = sorted(os.listdir(dataset_path))
    n_feats = []
    n_matches = []
    seq_type = []
    i_err = {thr: 0 for thr in rng}
    i_matches = {thr: 0 for thr in rng}
    v_err = {thr: 0 for thr in rng}
    v_matches = {thr: 0 for thr in rng}

    for seq_idx, seq_name in tqdm(enumerate(seq_names), total=len(seq_names)):
        keypoints_a, descriptors_a = read_feats(seq_name, 1)
        n_feats.append(keypoints_a.shape[0])

        for im_idx in range(2, 7):
            keypoints_b, descriptors_b = read_feats(seq_name, im_idx)
            n_feats.append(keypoints_b.shape[0])

            matches = mnn_matcher(
                torch.from_numpy(descriptors_a).to(device=device),
                torch.from_numpy(descriptors_b).to(device=device)
            )

            homography = np.loadtxt(os.path.join(dataset_path, seq_name, f"H_1_{im_idx}"))
            pos_a = keypoints_a[matches[:, 0], :2]
            pos_a_h = np.concatenate([pos_a, np.ones([matches.shape[0], 1])], axis=1)
            pos_b_proj_h = np.dot(homography, pos_a_h.T).T
            pos_b_proj = pos_b_proj_h[:, :2] / pos_b_proj_h[:, 2:]

            pos_b = keypoints_b[matches[:, 1], :2]
            dist = np.sqrt(np.sum((pos_b - pos_b_proj) ** 2, axis=1))

            n_matches.append(matches.shape[0])
            seq_type.append(seq_name[0])

            if dist.shape[0] == 0:
                dist = np.array([float("inf")])

            for thr in rng:
                if seq_name[0] == 'i':
                    i_err[thr] += np.mean(dist <= thr)
                    i_matches[thr] += np.sum(dist <= thr)
                else:
                    v_err[thr] += np.mean(dist <= thr)
                    v_matches[thr] += np.sum(dist <= thr)

    seq_type = np.array(seq_type)
    n_feats = np.array(n_feats)
    n_matches = np.array(n_matches)

    return i_err, v_err, i_matches, v_matches, [seq_type, n_feats, n_matches]

def summary(stats):
    seq_type, n_feats, n_matches = stats
    print('# Features: {:f} - [{:d}, {:d}]'.format(np.mean(n_feats), np.min(n_feats), np.max(n_feats)))
    print('# Matches: Overall {:f}, Illumination {:f}, Viewpoint {:f}'.format(
        np.sum(n_matches) / ((n_i + n_v) * 5),
        np.sum(n_matches[seq_type == 'i']) / (n_i * 5),
        np.sum(n_matches[seq_type == 'v']) / (n_v * 5))
    )

def generate_read_function(method, extension='ppm', type='float'):
    def read_function(seq_name, im_idx):
        subdir = f"{im_idx}.{extension}.{method}"
        kp_path = os.path.join(dataset_path, seq_name, subdir, "keypoints.npy")
        desc_path = os.path.join(dataset_path, seq_name, subdir, "descriptors.npy")
        keypoints = np.load(kp_path)
        descriptors = np.load(desc_path)
        if type == 'binary':
            descriptors = np.unpackbits(descriptors, axis=1, bitorder='little') * 2.0 - 1.0
        return keypoints, descriptors
    return read_function

errors = {}

for method in methods:
    print(method)
    if method == 'hesaff':
        read_function = lambda seq_name, im_idx: parse_mat(loadmat(os.path.join(dataset_path, seq_name, '%d.ppm.hesaff' % im_idx), appendmat=False))
    else:
        if method == 'delf' or method == 'delf-new':
            read_function = generate_read_function(method, extension='png')
        elif '+Boost-B' in method or method.lower() == 'orb':
            read_function = generate_read_function(method, type='binary')
        else:
            read_function = generate_read_function(method)

    errors[method] = benchmark_features(read_function)
    summary(errors[method][-1])

plt_lim = [1, 10]
plt_rng = np.arange(plt_lim[0], plt_lim[1] + 1)

plt.rc('axes', titlesize=25)
plt.rc('axes', labelsize=25)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
for method, name, color, ls in zip(methods, names, colors, linestyles):
    i_err, v_err, _, _, _ = errors[method]
    plt.plot(plt_rng, [(i_err[thr] + v_err[thr]) / ((n_i + n_v) * 5) for thr in plt_rng],
             color=color, ls=ls, linewidth=2, label=name)
plt.title('Overall')
plt.xlim(plt_lim)
plt.xticks(plt_rng)
plt.ylabel('MMA')
plt.ylim([0, 1])
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=20)
plt.legend()

plt.subplot(1, 3, 2)
for method, name, color, ls in zip(methods, names, colors, linestyles):
    i_err, _, _, _, _ = errors[method]
    plt.plot(plt_rng, [i_err[thr] / (n_i * 5) for thr in plt_rng],
             color=color, ls=ls, linewidth=2, label=name)
plt.title('Illumination')
plt.xlabel('threshold [px]')
plt.xlim(plt_lim)
plt.xticks(plt_rng)
plt.ylim([0, 1])
plt.gca().set_yticklabels([])
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=20)

plt.subplot(1, 3, 3)
for method, name, color, ls in zip(methods, names, colors, linestyles):
    _, v_err, _, _, _ = errors[method]
    plt.plot(plt_rng, [v_err[thr] / (n_v * 5) for thr in plt_rng],
             color=color, ls=ls, linewidth=2, label=name)
plt.title('Viewpoint')
plt.xlim(plt_lim)
plt.xticks(plt_rng)
plt.ylim([0, 1])
plt.gca().set_yticklabels([])
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=20)

plt.savefig('hseq.pdf', bbox_inches='tight', dpi=300)
