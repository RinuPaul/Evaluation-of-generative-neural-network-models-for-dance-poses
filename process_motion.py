"""
"""
import os
import argparse
import glob
from pathlib import Path
import numpy as np
from anim_utils.animation_data import BVHReader, MotionVector
from preprocessing.motion_features import load_skeleton, feature_vectors_from_motion
from preprocessing.utils import contains_element_in_list
from constants import IGNORE_LIST


def load_file(filename):
    bvh_reader = BVHReader(filename)
    mv = MotionVector()
    mv.from_bvh_reader(bvh_reader)
    return mv

def main(**kwargs):
    in_dir = kwargs["in_dir"]
    out_dir = kwargs["out_dir"]
    skeleton_file = kwargs["skeleton_file"]
    animated_joints = kwargs["animated_joints"]
    n_window_size = kwargs["n_window_size"]
    os.makedirs(out_dir, exist_ok=True)
    skeleton = load_skeleton(skeleton_file)
    if animated_joints is None:
        animated_joints = skeleton.animated_joints
    frame_time = skeleton.frame_time
    for f in Path(in_dir).iterdir():
        f = str(f)
        if not f.endswith("bvh") or contains_element_in_list(f, IGNORE_LIST):
            continue
        mv = load_file(f)
        _features = feature_vectors_from_motion(skeleton, mv.frames, animated_joints, frame_time)
        _features = np.array(_features)
        print(_features.shape)
        filename = out_dir + os.sep+ os.path.basename(f)[:-4]+ "_pc"
        print("save", filename)
        np.save(filename, _features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess motion data")
    parser.add_argument("--in_dir", type=str, default= r"data\test2\motions" )
    parser.add_argument("--out_dir", type=str, default= r"data\test2\motion_features")
    parser.add_argument("--skeleton_file", type=str, default=r"E:\Projects\CAROUSEL\workspace\preprocessing\data\smpl\no_offset\gBR_sBM_cAll_d06_mBR5_ch05.bvh")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--n_window_size", type=int, default=240)
    parser.add_argument("--animated_joints",  type=str, default=None)
    args = parser.parse_args()
    print(vars(args))
    main(**vars(args))

