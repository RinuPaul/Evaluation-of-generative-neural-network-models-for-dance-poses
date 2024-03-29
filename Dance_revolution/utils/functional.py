# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


""" Define the functions to load data. """
import os
import json
import argparse
import numpy as np

def save_poses(out_dir, poses, name_str):
    #name_str = "trail"
    out_filename = out_dir + os.sep + name_str + ".npz"
    os.makedirs(out_dir, exist_ok=True)
    #print("save ", len(poses)," poses to file ",out_filename)
    with open(out_filename, "wb") as out_file:
        np.save(out_file, poses)
        
def normalize(X_train):
    x_mean = np.mean(X_train, axis=0)
    x_std = np.std(X_train, axis=0)
    #X_train = (X_train-x_mean)/ (x_std+1e-8)
    X_train = (X_train-x_mean)/ (x_std)
    #print(len(X_train))
    return X_train

def load_data(data_dir, interval=100, data_type='3D'):
    music_data, dance_data = [], []
    fnames = sorted(os.listdir(data_dir))
    #print(fnames)
    # fnames = fnames[:10]  # For debug
    for fname in fnames:
        path = os.path.join(data_dir, fname)
        name = str(path)
        if name.split('/')[-1] != '.ipynb_checkpoints':
            with open(path) as f:
                sample_dict = json.loads(f.read())
                np_music = np.array(sample_dict['music_array'])
                np_dance = np.array(sample_dict['dance_array'])
                #print("shape",np_music.shape,np_dance.shape)
                if data_type == '2D':
                    # Only use 25 keypoints skeleton (basic bone) for 2D
                    np_dance = np_dance[:, :50]
                    root = np_dance[:, 2*8:2*9]
                    np_dance = np_dance - np.tile(root, (1, 25))
                    np_dance[:, 2*8:2*9] = root

                seq_len, dim = np_music.shape
                for i in range(0, seq_len, interval):
                    music_sub_seq = np_music[i: i + interval]
                    dance_sub_seq = np_dance[i: i + interval]
                    #print("seq_len",interval,music_sub_seq.shape)
                    if len(music_sub_seq) == interval:
                        music_data.append(music_sub_seq)
                        dance_data.append(dance_sub_seq)

    #print(len(music_data),len(dance_data))
    return music_data, dance_data


def load_test_data(data_dir, interval=100, data_type='3D'):
    music_data, dance_data = [], []
    fnames = sorted(os.listdir(data_dir))
    #print(fnames)
    # fnames = fnames[:60]  # For debug
    for fname in fnames:
        path = os.path.join(data_dir, fname)
        with open(path) as f:
            sample_dict = json.loads(f.read())
            np_music = np.array(sample_dict['music_array'])
            np_dance = np.array(sample_dict['dance_array'])
            print("shape",np_music.shape,np_dance.shape)
            if data_type == '2D':
                # Only use 25 keypoints skeleton (basic bone) for 2D
                np_dance = np_dance[:, :50]
                root = np_dance[:, 2*8:2*9]
                np_dance = np_dance - np.tile(root, (1, 25))
                np_dance[:, 2*8:2*9] = root

            seq_len, dim = np_music.shape
            for i in range(0, seq_len, interval):
                music_sub_seq = np_music[i: i + interval]
                dance_sub_seq = np_dance[i: i + interval]
                #print("seq_len",interval,music_sub_seq.shape)
                if len(music_sub_seq) == interval:
                    music_data.append(music_sub_seq)
                    dance_data.append(dance_sub_seq)
    #print(len(music_data),len(dance_data))

    return music_data, dance_data, fnames


def load_json_data(data_file, max_seq_len=150):
    music_data = []
    dance_data = []
    count = 0
    total_count = 0
    with open(data_file) as f:
        data_list = json.loads(f.read())
        for data in data_list:
            # The first and last segment may be unusable
            music_segs = data['music_segments']
            dance_segs = data['dance_segments']

            assert len(music_segs) == len(dance_segs), 'alignment'

            for i in range(len(music_segs)):
                total_count += 1
                if len(music_segs[i]) > max_seq_len:
                    count += 1
                    continue
                music_data.append(music_segs[i])
                dance_data.append(dance_segs[i])

    rate = count / total_count
    print(f'total num of segments: {total_count}')
    print(f'num of segments length > {max_seq_len}: {count}')
    print(f'the rate: {rate}')

    return music_data, dance_data


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
