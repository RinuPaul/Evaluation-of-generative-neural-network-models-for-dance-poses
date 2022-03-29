# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.

""" This script handling the test process. """
import os
import sys
import joblib 
import json
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data
from plot_util import extract_pose
from dataset import DanceDataset, paired_collate_fn
from utils.functional import str2bool, load_data, save_poses
from generator_baseline import Generator
from multiprocessing import Pool
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('--test_dir', type=str, default='../data/test_1min',
                    help='the directory of test data')
parser.add_argument('--data_type', type=str, default='3D',
                    help='the type of training data')
parser.add_argument('--output_dir', type=str, default='results/new_preprocess_ptcloud_rev',
                    help='the directory of generated result')
parser.add_argument('--cuda', type=str2bool, nargs='?', metavar='BOOL',
                    const=True, default=False,
                    help='whether to use GPU acceleration.')
parser.add_argument('--model', type=str, metavar='PATH',
                    default='results/new_preprocess_ptcloud_rev/epoch_1600.pt')
parser.add_argument('--batch_size', type=int, metavar='N', default=1)
parser.add_argument('--seq_len', type=int, metavar='N', default=200)

args = parser.parse_args()

def main():
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
       
    music_data, dance_data = load_data(
        args.test_dir, interval=args.seq_len, data_type=args.data_type
    )

    device = torch.device('cuda' if args.cuda else 'cpu')

    test_loader = torch.utils.data.DataLoader(
        DanceDataset(music_data, dance_data),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=paired_collate_fn,
        drop_last=True
    )

    generator = Generator(args, device)
    poses = []
    poses_in = []
    scale = 10
    n = 0
    for j in range(1):        
        if j == 0:
            frames_all = np.zeros((args.seq_len, 375), dtype=np.float32)
        else:
            frames_all2 = np.zeros((args.seq_len, 375), dtype=np.float32)
            
        for batch in tqdm(test_loader, desc='Generating dance poses'): # Prepare data
            src_seq, src_pos, tgt_seq = map(lambda x: x.to(device), batch)            
            tg_seq = tgt_seq[:, :10, :] # Choose the first 10 frames as the beginning
            pose_dims = tgt_seq.shape[-1]
            next_pos = generator.generate(src_seq, src_pos, tg_seq)
            n += 1 
            if j == 0:
                frames_all = next_pos[0]
            else:
                frames_all2 = next_pos[0]
#             break
#     cov_mat = np.cov(frames_all, frames_all2)
#     covariances = np.diag(cov_mat, frames_all.shape[0])
#     print('covariance:',np.sum(covariances)/args.seq_len)  
          
             # Visualize the generated dance poses  
            _y = np.array(frames_all)
            _z = tgt_seq.numpy()[0,:,:]
            _y *= scale
            _z *= scale
            n_frames = len(_y)
            for i in range(n_frames):
                y = _y[i]
                z = _z[i]
                pose = extract_pose(y, pose_dims)
                pose = np.array(pose).T
                poses.append(pose)
                pose_i = extract_pose(z, pose_dims)
                pose_i = np.array(pose_i).T
                poses_in.append(pose_i)
            save_poses(args.output_dir, np.array(poses, dtype = np.float32),"rev_dance"+str(n))
            save_poses(args.output_dir, np.array(poses_in, dtype = np.float32),"rev_dance_in"+str(n))             
        
if __name__ == '__main__':
    main()
