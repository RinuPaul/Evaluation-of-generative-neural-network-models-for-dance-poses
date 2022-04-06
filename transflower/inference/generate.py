import sys
import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(ROOT_DIR)
import numpy as np; import scipy.linalg
# LUL
w_shape = [219,219]
w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
np_p, np_l, np_u = scipy.linalg.lu(w_init)

from training.datasets import create_dataset, create_dataloader

from models import create_model
from training.options.train_options import TrainOptions
import torch
import pytorch_lightning as pl
import numpy as np
import pickle, json, yaml
import sklearn
import argparse
import os, glob
from pathlib import Path
from plot_util import extract_pose, save_poses
# from analysis.visualization.generate_video_from_mats import generate_video_from_mats
# from analysis.visualization.generate_video_from_expmaps import generate_video_from_expmaps
# from analysis.visualization.generate_video_from_moglow_pos import generate_video_from_moglow_loc

from training.utils import get_latest_checkpoint

if __name__ == '__main__':
    print("Hi")
    parser = argparse.ArgumentParser(description='Generate with model')
    parser.add_argument('--data_dir',default="data/motion", type=str)
    parser.add_argument('--seeds', type=str, help='in the format: mod,seq_id;mod,seq_id')
    parser.add_argument('--seeds_file', type=str, help='file from which to choose a random seed')
    parser.add_argument('--output_folder', type=str,default="inference/generated/")
    parser.add_argument('--audio_format', type=str, default="mp3")
    parser.add_argument('--experiment_name',default="transflower_expmap", type=str)
    parser.add_argument('--seq_id', type=str,default="gWA_sBM_cAll_d25_mWA2_ch05")
    parser.add_argument('--max_length', type=int, default=400)
    parser.add_argument('--no-use_scalers', dest='use_scalers', action='store_false')
    parser.add_argument('--generate_video', action='store_true')
    parser.add_argument('--generate_bvh', action='store_true')
    parser.add_argument('--generate_ground_truth', action='store_true')
    parser.add_argument('--fps', type=int, default=20)
    args = parser.parse_args()
    
    data_dir = args.data_dir
    audio_format = args.audio_format
    fps = args.fps
    print("video",args.generate_video)
    output_folder = args.output_folder
    seq_id = args.seq_id
    if args.seeds is not None:
        seeds = {mod:seq for mod,seq in [tuple(x.split(",")) for x in args.seeds.split(";")]}
    else:
        seeds = {}

    if seq_id is None:
        temp_base_filenames = [x[:-1] for x in open(data_dir + "/base_filenames_test.txt", "r").readlines()]
        seq_id = np.random.choice(temp_base_filenames)
    if args.seeds_file is not None:
        print("choosing random seed from "+args.seeds_file)
        temp_base_filenames = [x[:-1] for x in open(args.seeds_file, "r").readlines()]
        seq_id = np.random.choice(temp_base_filenames)


    print(seq_id)

    #load hparams file
    default_save_path = "training/experiments/"+args.experiment_name
    logs_path = default_save_path
    #latest_checkpoint = get_latest_checkpoint(logs_path)
    #print(latest_checkpoint)
    latest_checkpoint = "training/experiments/transflower_expmap/version_0/checkpoints/epoch=79-step=56999.ckpt"
    checkpoint_dir = Path(latest_checkpoint).parent.parent.absolute()
    # exp_opt = json.loads(open("training/experiments/"+args.experiment_name+"/opt.json","r").read())
    exp_opt = yaml.load(open(str(checkpoint_dir)+"/hparams.yaml","r").read(), Loader=yaml.FullLoader)
    opt = vars(TrainOptions().parse(parse_args=["--model", exp_opt["model"]]))
    #print(opt)
    opt.update(exp_opt)
    # opt["cond_concat_dims"] = True
    # opt["bn_momentum"] = 0.0
    opt["batch_size"] = 1
    opt["phase"] = "inference"
    opt["tpu_cores"] = 0
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    #print(opt)
    opt = Struct(**opt)

    input_mods = opt.input_modalities.split(",")
    output_mods = opt.output_modalities.split(",")
    output_time_offsets = [int(x) for x in str(opt.output_time_offsets).split(",")]
    if args.use_scalers:
        scalers = [x+"_scaler.pkl" for x in output_mods]
    else:
        scalers = []

    # Load latest trained checkpoint from experiment
    model = create_model(opt)
    model = model.load_from_checkpoint(latest_checkpoint, opt=opt)

    # Load input features (sequences must have been processed previously into features)
    features = {}
    for i,mod in enumerate(input_mods):
        if mod in seeds:
            feature = np.load(data_dir+"/"+seeds[mod]+"."+mod+".npy")
        else:
            feature = np.load(data_dir+"/"+seq_id+"."+mod+".npy")
            #print(feature.shape)
        if args.max_length != -1:
            feature = feature[:args.max_length]
        if model.input_fix_length_types[i] == "single":
            features["in_"+mod] = np.expand_dims(np.expand_dims(feature,1),1)
        else:
            features["in_"+mod] = np.expand_dims(feature,1)

    # Generate prediction
    if torch.cuda.is_available():
        model.cuda()
    #import pdb;pdb.set_trace()
    #import time
    #start_time = time.time()
    #print(feature.shape)
    #frames_all = np.zeros((args.max_length, 375), dtype=np.float32)
    for j in range(1):
        predicted_mods = model.generate(features, ground_truth=args.generate_ground_truth)
        if j == 0:
            frames = predicted_mods[0]
            frames_all = frames.cpu().numpy()[:,0,:]
        else:
            frames2 = predicted_mods[0]
            frames_all2 = frames2.cpu().numpy()[:,0,:]
#     #vectorized
#     cov_mat = np.cov(frames_all, frames_all2)
#     covariances = np.diag(cov_mat, frames_all.shape[0])
#     #print(np.max(covariances))
#     print('covariance:',np.sum(covariances)/frames_all.shape[0])
    poses = []
#     poses_in = []
    scale = 7
    pose_dims = 375
    frames_all = predicted_mods[0]
    #print(frames_all.shape)
    _y = frames_all.cpu().numpy()[:,0,:]
#     _z = features["in_"+"_pc"][:_y.shape[0],0,:]
    _y *= scale
#     _z *= scale
#     print(_y.shape, _z.shape)
    n_frames = len(_y)
    for i in range(n_frames):
        y = _y[i]
#         z = _z[i]
        pose = extract_pose(y, pose_dims)
        pose = np.array(pose).T
        poses.append(pose)
#         pose_i = extract_pose(z, pose_dims)
#         pose_i = np.array(pose_i).T
#         poses_in.append(pose_i)
    save_poses(args.output_folder, np.array(poses, dtype = np.float32),"tf_gen_5")
#     save_poses(args.output_folder, np.array(poses_in, dtype = np.float32),"tf_in_1")  
    print("Done generating")

# python inference/generate.py