# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.
'''python3.7 train.py'''

""" This script handling the training process. """
import os
import sys
import time
import math
import random
import joblib
import pickle
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
from dataset import DanceDataset, paired_collate_fn
from model_baseline import Encoder, Decoder, Model
from plot_util import extract_pose
from simple_regression_torch import create_simple_regression_model
from utils.log import Logger
from utils.functional import str2bool, load_data, normalize, save_poses
import warnings
warnings.filterwarnings('ignore')
module_path = os.path.abspath(os.path.join('./analysis'))
if module_path not in sys.path:
    sys.path.append(module_path)
from pymo.writers import *

def train(model, training_data, optimizer, device, args, log):
    """ Start training """
    #criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    #criterion = nn.KLDivLoss(size_average=False)
    updates = 0  # global step
    loss = 0

    for epoch_i in range(1, args.epoch + 1):
        log.set_progress(epoch_i, len(training_data))
        model.train()
        if math.isnan(loss):
            break

        for batch_i, batch in enumerate(training_data):
            
            # prepare data
            src_seq, src_pos, tgt_seq = map(lambda x: x.to(device), batch)
            #print(tgt_seq.shape,src_seq.shape,src_pos.shape)
            gold_seq = tgt_seq[:, 1:]
            src_pos = src_pos[:, :-1]         
            src_seq = src_seq[:, :-1]   
            tgt_seq = tgt_seq[:, :-1]          
            
            hidden, out_frame, out_seq = model.module.init_decoder_hidden(tgt_seq.size(0))
            # forward
            optimizer.zero_grad()

            output = model(src_seq, src_pos, tgt_seq, hidden, out_frame, out_seq, epoch_i)        
                        
            # backward
            #loss = Glow.loss_generative(nll)#
            loss = criterion(output, gold_seq)
            loss.backward()

            # update parameters
            optimizer.step()

            stats = {
                'updates': updates,
                'loss': loss.item()
            }
            log.update(stats)
            updates += 1

        checkpoint = {
            'model': model.state_dict(),
            'args': args,
            'epoch': epoch_i
        }
        if epoch_i%args.save_per_epochs == 0:
            poses = []
            scale=10
            pose_dims = 375 #95
            _y = output.cpu().data.numpy()[0,:,:]
            _y *= scale
            n_frames = len(_y) #int(len(_y)/pose_dims)
            for i in range(n_frames):
                y = _y[i]
                pose = extract_pose(y, pose_dims, o=3) #  taking only positions 3-95 in 375 features
                pose = np.array(pose).T
                poses.append(pose)
            name = 'train'+str(epoch_i)
            save_poses(args.output_dir, np.array(poses),name)
         
        if epoch_i % args.save_per_epochs == 0 and not math.isnan(loss):
            filename = os.path.join(args.output_dir, f'epoch_{epoch_i}.pt')
            torch.save(checkpoint, filename)


def main():
    """ Main function """
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', type=str, default='../data/train_1min',
                        help='the directory of dance data')
    parser.add_argument('--test_dir', type=str, default='../data/test_1min',
                        help='the directory of music feature data')
    parser.add_argument('--data_type', type=str, default='3D',
                        help='the type of training data')
    parser.add_argument('--output_dir', metavar='PATH',
                        default='results/new_preprocess_ptcloud_rev_flow')

    parser.add_argument('--epoch', type=int, default=160)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_per_epochs', type=int, metavar='N', default=20)
    parser.add_argument('--log_per_updates', type=int, metavar='N', default=1,
                        help='log model loss per x updates (mini-batches).')
    parser.add_argument('--seed', type=int, default=1234,
                       help='random seed for data shuffling, dropout, etc.')
    parser.add_argument('--tensorboard', action='store_false')

    parser.add_argument('--d_frame_vec', type=int, default=27)
    parser.add_argument('--frame_emb_size', type=int, default=200)
    parser.add_argument('--d_pose_vec', type=int, default=375)
    parser.add_argument('--pose_emb_size', type=int, default=200)

    parser.add_argument('--d_inner', type=int, default=1024)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.6)

    parser.add_argument('--seq_len', type=int, default=200)
    parser.add_argument('--max_seq_len', type=int, default=200)
    parser.add_argument('--condition_step', type=int, default=10)
    parser.add_argument('--sliding_windown_size', type=int, default=100)
    parser.add_argument('--lambda_v', type=float, default=0.1)

    parser.add_argument('--cuda', type=str2bool, nargs='?', metavar='BOOL', const=True,
                        default=torch.cuda.is_available()
                        ,help='whether to use GPU acceleration.')

    args = parser.parse_args()
    args.d_model = args.frame_emb_size

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    global log
    log = Logger(args)
    print(args)
    # Set random seed
    device = torch.device('cuda' if args.cuda else 'cpu')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Loading training data
    train_music_data, train_dance_data = load_data(
        args.train_dir, 
        interval=args.seq_len,
        data_type=args.data_type)
    
    training_data = prepare_dataloader(train_music_data, train_dance_data, args)

    encoder = Encoder(max_seq_len=args.max_seq_len,
                      input_size=args.d_frame_vec,
                      d_word_vec=args.frame_emb_size,
                      n_layers=args.n_layers,
                      n_head=args.n_head,
                      d_k=args.d_k,
                      d_v=args.d_v,
                      d_model=args.d_model,
                      d_inner=args.d_inner,
                      dropout=args.dropout)

    decoder = Decoder(input_size=args.d_pose_vec,
                      d_word_vec=args.pose_emb_size,
                      hidden_size=args.d_inner,
                      encoder_d_model=args.d_model,
                      dropout=args.dropout)
    
    model = Model(encoder, decoder, 
                  condition_step=args.condition_step,
                  sliding_windown_size=100,
                  lambda_v=args.lambda_v,
                  device=device)

#     pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(pytorch_total_params)
    
    # Data Parallel to use multi-gpu
    model = nn.DataParallel(model).to(device)

    optimizer = optim.Adam(filter(
        #lambda x: x.requires_grad, model.parameters()), weight_decay=1e-5, lr=args.lr)
        lambda x: x.requires_grad, model.module.parameters()),  lr=args.lr, weight_decay=1e-5)

    train(model, training_data, optimizer, device, args, log)


def prepare_dataloader(music_data, dance_data, args):
    data_loader = torch.utils.data.DataLoader(
        DanceDataset(music_data, dance_data),
        num_workers=0,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=paired_collate_fn
    )

    return data_loader


if __name__ == '__main__':
    main()
