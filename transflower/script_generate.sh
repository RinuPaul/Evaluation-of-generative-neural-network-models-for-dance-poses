#!/bin/bash



py=python3

#exp=transglower_residual_aistpp_expmap
#exp=moglow_aistpp_expmap
exp='transglower'
#seq_id=gKR_sFM_cAll_d28_mKR5_ch06
#seq_id=gLH_sFM_cAll_d16_mLH3_ch04
#seq_id=aistpp_gPO_sFM_cAll_d12_mPO4_ch19
#seq_id=aistpp_gMH_sFM_cAll_d22_mMH3_ch04
#seq_id=$2
#seq_id=Streetdance_001
#seq_id=shadermotion_7
#echo $exp $seq_id

mkdir inference/generated/
mkdir inference/generated/${exp}
mkdir inference/generated/${exp}/predicted_mods
mkdir inference/generated/${exp}/videos
fps=20
#data_dir=data/aistpp_20hz
#data_dir=$SCRATCH/data/aistpp_20hz
#data_dir=$SCRATCH/data/dance_combined
#data_dir=$SCRATCH/data/dance_combined_test
#data_dir=$SCRATCH/data/dance_combined2_test
#data_dir=$SCRATCH/data/dance_combined2
#data_dir=test_data
#data_dir=$SCRATCH/data/moglow_pos

# if we don't pass seq_id it will choose a random one from the test set
$py inference/generate.py --generate_video --generate_bvh 


