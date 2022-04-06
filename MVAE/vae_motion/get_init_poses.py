import os
import sys
import numpy as np



def get_init_pose_for_AIST():

    data_folder = r'D:\workspace\my_git_repos\mi_thesis_variational_dance_motion_models'
    hp_gan_training_data = np.load(os.path.join(data_folder, 'AIST_training_data_HPGAN.npy'))
    print(hp_gan_training_data.shape)
    init_pose = hp_gan_training_data[0, 0:1, :]
    print(init_pose.shape)

    current_dirname = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_dirname, '..', 'environments')
    np.save(os.path.join(save_path, 'AIST_music.npy'), init_pose)



if __name__ == "__main__":
    get_init_pose_for_AIST()