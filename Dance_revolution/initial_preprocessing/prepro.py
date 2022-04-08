import os
import sys
import argparse
import numpy as np
import glob
import librosa
import random
import json
from process_motion_v2 import process_data

parser = argparse.ArgumentParser()
parser.add_argument('--input_dance_dir', type=str, default='data/test2/AIST_main')
parser.add_argument('--train_dir', type=str, default='data/train_1min')
parser.add_argument('--test_dir', type=str, default='data/test_1min')
args = parser.parse_args()

#extractor = FeatureExtractor()

if not os.path.exists(args.train_dir):
    os.mkdir(args.train_dir)
if not os.path.exists(args.test_dir):
    os.mkdir(args.test_dir)

def extract_feature():
    dances = []
    musics = []
    data_folder = r'data/test2'
    audio_path = os.path.join(data_folder, 'aistmusic')
    motion_path = os.path.join(data_folder, 'AIST_main')
    #print(audio_path,motion_path)
    bvhfiles = glob.glob(os.path.join(motion_path, '*.bvh'))
    print(len(bvhfiles))
    data_clips = []
    i=0
    for bvhfile in bvhfiles:
        print(i)
        filename = os.path.split(bvhfile)[-1]
        audio_file = os.path.join(audio_path, filename.replace('.bvh', '.mp3'))
        if not os.path.exists(audio_file):
            raise FileExistsError("Cannot find audio file")        

        motion_data = extract_motion_features(bvhfile)
        audio_data = extract_audio_features(audio_file, len(motion_data))
        musics.append(audio_data.tolist())
        print(audio_data.shape)
        dances.append(motion_data.tolist())
        print("dance_dim",motion_data.shape)
        i+=1
    
    return musics, dances ,bvhfiles
    

def extract_motion_features(bvhfile):
    """use H36M skeleton
    Args:
        bvhfile ([type]): [description]
    Returns:
        [type]: [description]
    """
    torso_joints = ['LeftUpLeg', 'LowerBack', 'RightUpLeg']
    left_foot_joints = ['LeftFoot', 'LeftToeBase']
    right_foot_joints = ['RightFoot', 'RightToeBase']
    motion_data = process_data(bvhfile, torso_joints, left_foot_joints, right_foot_joints, sliding_window=False)
    return motion_data



def extract_audio_features(audio_file, n_frames):

    audio_data, sr = librosa.load(audio_file)
    hop_len = np.floor(len(audio_data) / n_frames)
    C = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_fft=2048, 
                                    hop_length=int(hop_len), n_mels=27, fmin=0.0, fmax=8000)
    
    return C.T[:n_frames]


def segment_data(input_sequence, window=60, window_step=30):
    windows = []
    # windows_classes = []
    if len(input_sequence) % window_step == 0:
        n_clips = (len(input_sequence) - len(input_sequence) % window_step)//window_step 
    else:
        n_clips = (len(input_sequence) - len(input_sequence) % window_step) // window_step + 1
    for j in range(0, n_clips):
        """ If slice too small pad out by repeating start and end poses """
        slice = input_sequence[j * window_step : j * window_step + window]
        if len(slice) < window:
            left = slice[:1].repeat((window - len(slice)) // 2 + (window - len(slice)) % 2, axis=0)
            right = slice[-1:].repeat((window - len(slice)) // 2, axis=0)
            slice = np.concatenate([left, slice, right], axis=0)
        if len(slice) != window: raise Exception()
        windows.append(slice)
    return windows

def align(musics, dances):
    print('---------- Align the frames of music and dance ----------')
    print(len(musics),len(dances))
    assert len(musics) == len(dances), \
        'the number of audios should be equal to that of videos'
    new_musics=[]
    new_dances=[]
    for i in range(len(musics)):
        min_seq_len = len(musics[i])
        new_musics.append([musics[i][j] for j in range(min_seq_len) if j%2==0])
        new_musics.append([musics[i][j] for j in range(min_seq_len) if j%2==1])
        
        new_dances.append([dances[i][j] for j in range(min_seq_len) if j%2==0])
        new_dances.append([dances[i][j] for j in range(min_seq_len) if j%2==1])
    return new_musics, new_dances

def split_data(fnames):
    print('---------- Split data into train and test ----------')
    indices = list(range(len(fnames)))
    random.shuffle(indices)
    train = indices[:]
    #test = indices[:5]


    return train#, test

def save(args, musics, dances, bvhfiles):
    print('---------- Save to text file ----------')
    fnames = bvhfiles#sorted(os.listdir(args.input_dance_dir))
    print(len(musics),len(dances))
    assert len(fnames)*2 == len(musics) == len(dances), 'alignment'

    train_idx = split_data(fnames)
    #test_idx = split_data(fnames)
    train_idx = sorted(train_idx)
    #print(f'train ids: {[fnames[idx] for idx in train_idx]}')
    #test_idx = sorted(test_idx)
    #print(f'test ids: {[fnames[idx] for idx in test_idx]}')

    print('---------- train data ----------')

    for idx in train_idx:
        for i in range(2):
            nameo = os.path.split(fnames[idx])[-1]
            name = nameo.split('.')[0]
            with open(os.path.join(args.train_dir, f'{name+"_"+str(i)}.json'), 'w') as f:
                sample_dict = {
                    'id': name+"_"+str(i),
                    'music_array': musics[idx*2+i],
                    'dance_array': dances[idx*2+i]
                }
                json.dump(sample_dict, f)

    '''print('---------- test data ----------')
    for idx in test_idx:
        for i in range(2):
            name = fnames[idx].split('.')[0]
            with open(os.path.join(args.test_dir, f'{name+"_"+str(i)}.json'), 'w') as f:
                sample_dict = {
                    'id': name+"_"+str(i),
                    'music_array': musics[idx*2+i],
                    'dance_array': dances[idx*2+i]
                }
                json.dump(sample_dict, f)'''


if __name__ == "__main__":
    
    musics, dances, bvhfiles = extract_feature() 
    musics, dances = align(musics, dances)
    save(args, musics, dances, bvhfiles)
