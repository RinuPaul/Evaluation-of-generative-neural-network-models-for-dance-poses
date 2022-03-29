import os
import sys
import numpy as np
import glob
import librosa
from process_motion_v2 import process_data


def create_training_data_for_HPGAN():
    data_folder = r'data\transflower'
    audio_path = os.path.join(data_folder, 'AIST_music')
    motion_path = os.path.join(data_folder, 'AIST_motion_retargeted')
    bvhfiles = glob.glob(os.path.join(motion_path, '*.bvh'))
    data_clips = []
    for bvhfile in bvhfiles:
        filename = os.path.split(bvhfile)[-1]
        audio_file = os.path.join(audio_path, filename.replace('.bvh', '.mp3'))
        if not os.path.exists(audio_file):
            raise FileExistsError("Cannot find audio file")
        

        motion_data = extract_motion_features(bvhfile)
        audio_data = extract_audio_features(audio_file, len(motion_data))
        combined_features = np.concatenate((motion_data, audio_data), axis=-1)
        segmented_data = segment_data(combined_features)
        data_clips += segmented_data
    data_clips = np.asarray(data_clips)
    print(data_clips.shape)
    np.save('AIST_training_data_HPGAN.npy', data_clips)


def create_training_data_for_mvae():
    data_folder = r'data\transflower'
    audio_path = os.path.join(data_folder, 'AIST_music')
    motion_path = os.path.join(data_folder, 'AIST_motion_retargeted')
    bvhfiles = glob.glob(os.path.join(motion_path, '*.bvh'))
    data_clips = []    
    end_indices = []
    current_index = -1
    for bvhfile in bvhfiles:
        filename = os.path.split(bvhfile)[-1]
        audio_file = os.path.join(audio_path, filename.replace('.bvh', '.mp3'))
        if not os.path.exists(audio_file):
            raise FileExistsError("Cannot find audio file")
        motion_data = extract_motion_features(bvhfile)
        audio_data = extract_audio_features(audio_file, len(motion_data))
        combined_features = np.concatenate((motion_data, audio_data), axis=-1)           
        current_index += len(combined_features)
        end_indices.append(current_index)
        data_clips.append(combined_features)
    training_data = np.concatenate(data_clips, axis=0)
    np.savez_compressed('AIST_music_mvae.npz', data=training_data, end_indices=end_indices)


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



if __name__ == "__main__":
    # create_training_data()
    create_training_data_for_mvae()
