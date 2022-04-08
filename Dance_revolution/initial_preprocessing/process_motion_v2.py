import os
import sys
import glob
import numpy as np
import scipy.ndimage.filters as filters
import argparse
from pathlib import Path
from transformations import *
from mi_anim_utils.animation_data import BVHReader, SkeletonBuilder
from mi_anim_utils.animation_data.utils import convert_euler_frames_to_cartesian_frames
from mi_anim_utils.animation_data.body_plane import BodyPlane
from mi_anim_utils.animation_data.quaternion import Quaternion
from preprocessing.utils import contains_element_in_list
from constants import IGNORE_LIST


def get_rotation_to_ref_direction(dir_vecs, ref_dir):
    rotations = []
    for dir_vec in dir_vecs:
        rotations.append(Quaternion.between(dir_vec, ref_dir))
    return rotations


def softmax(x, **kw):
    softness = kw.pop('softness', 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))


def softmin(x, **kw):
    return -softmax(-x, **kw)


def get_files(dir, suffix=".bvh"):
    files = []
    for f in Path(dir).iterdir():
        f = str(f)
        if  f.endswith(suffix) and not contains_element_in_list(f, IGNORE_LIST):
            files.append(f)
    return files
        

def cartesian_pose_orientation(cartesian_pose, body_plane_index, up_axis):
    assert len(cartesian_pose.shape) == 2
    up_axis = np.asarray(up_axis)
    points = cartesian_pose[body_plane_index, :]
    body_plane = BodyPlane(points)
    normal_vec = body_plane.normal_vector
    normal_vec[np.where(up_axis == 1)] = 0  ### only consider forward direction on the ground
    return normal_vec/np.linalg.norm(normal_vec)


def get_forward_direction(global_positions):
    ### only for mk_cmu_skeleton
    sdr_l, sdr_r, hip_l, hip_r = 10, 20, 2, 27
    across = (
        (global_positions[:,sdr_l] - global_positions[:,sdr_r]) + 
        (global_positions[:,hip_l] - global_positions[:,hip_r]))
    across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]
    """ Smooth Forward Direction """
    
    direction_filterwidth = 20
    forward = filters.gaussian_filter1d(
        np.cross(across, np.array([[0,1,0]])), direction_filterwidth, axis=0, mode='nearest')    
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]        
    return forward


def rotate_vectors(vectors, q):
    '''
    rotate a cartesian frame by given quaternion q
    :param vectors: ndarray (n_joints * 3)
    :param q: Quaternion
    :return:
    '''
    new_cartesian_frame = np.zeros(vectors.shape)
    for i in range(len(vectors)):
        new_cartesian_frame[i] = q * vectors[i]
    return new_cartesian_frame


def process_data(filename, 
                torso_joints, 
                left_foot_joints,
                right_foot_joints,
                animated_joints=None,
                window=240, 
                window_step=120, 
                sliding_window=True,
                forward_axis=[0, 0, 1],
                up_axis=[0, 1, 0]):
    """ Compute joint positions for animated joints """
    """ Motion VAE paper format
    """
    print(filename)
    bvhreader = BVHReader(filename)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    n_frames = len(bvhreader.frames)
    print("Number of frame is: {}".format(n_frames))
    if animated_joints is None:
        animated_joints = skeleton.animated_joints

    n_joints = len(animated_joints)
    torso_indices = [animated_joints.index(joint) for joint in torso_joints]
    left_foot_indices = [animated_joints.index(joint) for joint in left_foot_joints]
    right_foot_indices = [animated_joints.index(joint) for joint in right_foot_joints]
    #print(left_foot_indices,right_foot_indices)

    global_positions = convert_euler_frames_to_cartesian_frames(skeleton, bvhreader.frames,
                                                                animated_joints=animated_joints)  
 
    last_pose = global_positions[-1, :, :]                                                
    global_positions = np.append(global_positions, last_pose[np.newaxis, :, :], axis=0)
                                                                                                           
    """ Get Global Rotation """
    global_orientation = np.zeros((n_frames, n_joints, 4, 4))
    for i in range(n_frames):
        for j in range(n_joints):
            
            global_orientation[i, j] = skeleton.nodes[animated_joints[j]].get_global_matrix_from_euler_frame(bvhreader.frames[i])
        
    """ use forward and upward direction vectors to representation orientation """
      
    global_forward_vectors = global_orientation[:, :, :3, 2]
    global_up_vectors = global_orientation[:, :, :3, 1]
    
    forward = []
    for i in range(len(global_positions)):
        forward.append(cartesian_pose_orientation(global_positions[i], torso_indices, up_axis))
    forward = np.asarray(forward)
    rotations = get_rotation_to_ref_direction(forward, ref_dir=forward_axis)

    """ Put on Floor """

    foot_heights = np.minimum(global_positions[:, left_foot_indices, 1], global_positions[:, right_foot_indices, 1]).min(axis=1)
    floor_height = softmin(foot_heights, softness=0.5, axis=0)
    #print(foot_heights[1])
    ### first, put on the ground
    global_positions = global_positions - floor_height
    print(global_positions[:10,5,:],global_positions[:10,30,:])

    """ Get Root Velocity """
    root_velocity = global_positions[1:, 0:1] - global_positions[:-1, 0:1]

    """ Remove Translation """
    ### second remove translation in x and z axis
    local_positions = global_positions.copy()
    local_positions[:, :, 0] = global_positions[:, :, 0] - global_positions[:, 0:1, 0]
    local_positions[:, :, 2] = global_positions[:, :, 2] - global_positions[:, 0:1, 2]
    local_velocities = global_positions[1:] - global_positions[:-1]

    local_forward_vectors = global_forward_vectors.copy()
    local_up_vectors = global_up_vectors.copy()
    """ Remove Y Rotation """
    ### thirdly, rotated frames
    for i in range(n_frames):
        local_positions[i] = rotate_vectors(local_positions[i], rotations[i])
        local_velocities[i] = rotate_vectors(local_velocities[i], rotations[i])
        local_forward_vectors[i] = rotate_vectors(local_forward_vectors[i], rotations[i])
        local_up_vectors[i] = rotate_vectors(local_up_vectors[i], rotations[i])

    """ Rotate Velocity """
    for i in range(n_frames):
        root_velocity[i, 0] = rotations[i] * root_velocity[i, 0]  ### root velocity shape: (n_frames * 1 * 3)
    """ Get Rotation Velocity """
    r_v = np.zeros(n_frames)
    for i in range(n_frames):
        q = rotations[i+1] * (-rotations[i])   ### rotations[0] is not identical
        r_v[i] = Quaternion.get_angle_from_quaternion(q, np.asarray(forward_axis))
    local_positions = local_positions[:-1]
    """ Add Root_Velocity, RVelocity, Foot Contacts to vector """
    print(local_positions[:10,5,:],local_positions[:10,30,:] )
    h
    #print(local_positions.shape)
    #print(local_velocities.shape)
    #print(local_forward_vectors.shape)
    #print(local_up_vectors.shape)
    output_features = np.concatenate([root_velocity[:, :, 0], root_velocity[:, :, 2], r_v[:, np.newaxis]], axis=-1)
    #print(output_features.shape)
    output_features = np.append(output_features, local_positions.reshape(n_frames, -1), axis=-1)  ## joint position
    #print(output_features.shape)
    output_features = np.append(output_features, local_velocities.reshape(n_frames, -1), axis=-1)  ## joint velocity
    #print(output_features.shape)
    output_features = np.append(output_features, local_forward_vectors.reshape(n_frames, -1), axis=-1)  ## joint forward directions
    #print(output_features.shape)
    output_features = np.append(output_features, local_up_vectors.reshape(n_frames, -1), axis=-1) ## joint up direction
    #print(output_features.shape)
    

    if sliding_window:
        """ Slide Over Windows """
        windows = []
        # windows_classes = []
        if len(output_features) % window_step == 0:
            n_clips = (len(output_features) - len(output_features) % window_step)//window_step 
        else:
            n_clips = (len(output_features) - len(output_features) % window_step) // window_step + 1
        for j in range(0, n_clips):
            """ If slice too small pad out by repeating start and end poses """
            slice = output_features[j * window_step : j * window_step + window]
            if len(slice) < window:
                left = slice[:1].repeat((window - len(slice)) // 2 + (window - len(slice)) % 2, axis=0)
                left[:, -7:-4] = 0.0
                right = slice[-1:].repeat((window - len(slice)) // 2, axis=0)
                right[:, -7:-4] = 0.0
                slice = np.concatenate([left, slice, right], axis=0)
            if len(slice) != window: raise Exception()

            windows.append(slice)
        return windows

    else:
        return output_features



def main(**kwargs):
    data_folder = kwargs["in_dir"]
    save_path = kwargs["out_dir"]
    compress = kwargs.get("compress", False)
    os.makedirs(save_path, exist_ok=True)
    proportion = 1.0#0.1
    bvhfiles = get_files(data_folder, ".bvh")
    bvhfiles = sorted(bvhfiles)
    datasize = int(len(bvhfiles) * proportion)
    end_indices = []
    motion_clips = []
    torso_joints = ['3', '2', '1']
    left_foot_joints = ['7', '10']
    right_foot_joints = ['8', '11']  
    forward_axis = np.array([0, 0, 1])
    current_index = -1
    for bvhfile in bvhfiles[:datasize]:
        motion_clip = process_data(bvhfile,
                                #    animated_joints=MH_CMU_ANIMATED_JOINTS,
                                    torso_joints=torso_joints,
                                    left_foot_joints=left_foot_joints,
                                    right_foot_joints=right_foot_joints,
                                    sliding_window=False)
        if not compress:
            filename = save_path + os.sep+ os.path.basename(bvhfile)[:-4]+ "_pc"
            np.save(filename, motion_clip)
        current_index += len(motion_clip)
        motion_clips.append(motion_clip)

        end_indices.append(current_index)

    if compress:
        dataset_name = os.path.split(data_folder)[-1]
        motion_clips = np.concatenate(motion_clips, axis=0)
        if proportion != 1:
            save_filename = dataset_name + '_' + str(proportion) + '.npz'
        else:
            save_filename = dataset_name + '.npz'
        np.savez_compressed(os.path.join(save_path, save_filename), data=motion_clips, end_indices=end_indices)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess motion data")
    parser.add_argument("--in_dir", type=str, default= r"data\test2\motion" )
    parser.add_argument("--out_dir", type=str, default= r"data\test2\motion_features")
    parser.add_argument("--skeleton_file", type=str, default=r"data\test2\AIST_main\gBR_sBM_cAll_d06_mBR5_ch05.bvh")
    args = parser.parse_args()
    print(vars(args))
    main(**vars(args))