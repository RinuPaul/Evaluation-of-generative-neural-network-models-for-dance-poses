import os
import numpy as np 
from src_final.model.HPGAN import HPGAN
from mosi_dev_deepmotionmodeling.mosi_utils_anim import load_json_file, write_to_json_file
from mosi_dev_deepmotionmodeling.utilities.utils import export_point_cloud_data_without_foot_contact
from mosi_dev_deepmotionmodeling.mosi_utils_anim.animation_data import BVHReader, SkeletonBuilder
from mosi_dev_deepmotionmodeling.preprocessing.preprocessing import process_file
from src_final.model.HPGAN import HPGAN
from src_final.model.BIHMPGAN import BIHMPGAN
from src_final.network.Discriminator import HPDiscriminator, Bihmp_Discriminator, CNN_Discriminator
from src_final.network.Generator import HPGenerator, Bihmp_Generator, TCN_RNN_Generator
from src_final.model.motionGAN_recon import motionGAN_recon
from src_final.model.motionGAN_norecon import motionGAN_norecon
from src_final.utils import CMU_SKELETON, CMU_ANIMATED_JOINTS, CMU_JOINT_LEVELS
from src_final.network.PoseEmbedding import SpatialEncoder, SpatialDecoder
from src_final.utils import CMU_SKELETON, 
EPS = 1e-6
TEST_DS = ['art','captury','optitrack','vicon']

def preprocess(filename, window):
    clip = process_file(filename, window=window, window_step=window, sliding_window=True, animated_joints=CMU_ANIMATED_JOINTS,
                 body_plane_indices=[6,27,2], fid_l=[4,5], fid_r=[29,30])
    return clip

def get_config(model_name, config_id):
    config_filepath = os.path.join('config', model_name, "config_" + str(config_id) + ".json")
    return load_json_file(config_filepath)

def get_test_data_h36m():
    training_data = np.load(r'data/training_data/training_clips_h36m.npy')
    print(training_data.shape)
    mean_pose = training_data.mean(axis=(0, 1))
    std_pose = training_data.std(axis=(0, 1))
    std_pose[std_pose<EPS] = EPS   
    test_set = training_data[int(0.9*len(training_data)):]
    return test_set, mean_pose, std_pose    

def get_test_data_motioncapture(dsname, seq_len):
    dataset_dir = 'data/training_data/preprocessed/input_data/preprocessed'
    if dsname == 'art':
        test_set = os.path.join(dataset_dir, 'ART_6kmh.bvh')
    if dsname == 'captury':
        test_set = os.path.join(dataset_dir, 'Captury_6kmh.bvh')
    if dsname == 'optitrack':
        test_set = os.path.join(dataset_dir, 'OptiTrack_6kmh.bvh')
    if dsname == 'vicon':
        test_set = os.path.join(dataset_dir, 'Vicon_6kmh.bvh')
    test_data = preprocess(test_set, seq_len)
    test_data = np.asarray(test_data, dtype=np.float32)
    mean_pose = test_data.mean(axis=(0, 1))
    std_pose = test_data.std(axis=(0, 1))
    std_pose[std_pose<EPS] = EPS
    return test_data, mean_pose, std_pose

def evaluate_HPGAN_motion():
    modelname = "HP-GAN"
    config_id = 0
    epoch = 200
    name = modelname + '_' + str(config_id)
    config = get_config(modelname, config_id)
    ## model initialization
    g = HPGenerator(config["generator"])
    d = HPDiscriminator(config["discriminator"])
    e = HPDiscriminator(config["evaluator"])
    model = HPGAN(g, d, e, name, debug=True)
    model.load(suffix=str(200))

    ## load h36m test data
    input_sequence_length = config['data']['input_sequence_length']
    test_set, mean_pose, std_pose = get_test_data_h36m()
    test_sequence = test_set[:, :input_sequence_length, :]
    test_sequence = (test_sequence - mean_pose) / std_pose 
    ## motion prediction
    model.sample(test_sequence, test_set, std_pose, mean_pose, config['train']["bvh_file"], epoch=epoch)
    ## load 4 motion capture test data
    for dsname in TEST_DS:
        test_data, mean_pose, std_pose = get_test_data_motioncapture(dsname, config['data']['input_sequence_length']+config['data']['output_sequence_length'])
        test_sequence = test_data[:, :input_sequence_length, :]
        test_sequence = (test_sequence - mean_pose) / std_pose
        model.sample(test_sequence, test_data, std_pose, mean_pose, config['train']["bvh_file"], epoch=epoch, sample_path=os.path.join('output', name, 'sample_'+dsname))

def evaluate_HPGAN_quality():
    modelname = "HP-GAN"
    config_id = 0
    epoch = 200
    name = modelname + '_' + str(config_id)
    config = get_config(modelname, config_id)
    ## model initialization
    g = HPGenerator(config["generator"])
    d = HPDiscriminator(config["discriminator"])
    e = HPDiscriminator(config["evaluator"])
    model = HPGAN(g, d, e, name, debug=True)
    model.load(suffix=str(200))

    ## load h36m test data
    input_sequence_length = config['data']['input_sequence_length']
    test_set, mean_pose, std_pose = get_test_data_h36m()
    test_set = (test_set - mean_pose) / std_pose 
    ## training data quality
    training_quality = model.e(test_set)
    if type(training_quality) is tuple:
        training_quality = np.mean(training_quality[0].numpy())
    else:
        training_quality = np.mean(training_quality.numpy()) 
    ## load 4 motion capture test data
    test_qualities = []
    for dsname in TEST_DS:
        test_data, mean_pose, std_pose = get_test_data_motioncapture(dsname, config['data']['input_sequence_length']+config['data']['output_sequence_length'])
        test_data = (test_data - mean_pose) / std_pose
        test_quality = model.e(test_data)
        if type(test_quality) is tuple:
            test_quality = np.mean(test_quality[0].numpy())
        else:
            test_quality = np.mean(test_quality.numpy())
        test_qualities.append(test_quality)
    quality = os.path.join('output', name, 'quality.npz')
    np.savez(quality, h36m=training_quality, art=test_qualities[0], captury=test_qualities[1], optitrack=test_qualities[2], vicon=test_qualities[3])

def evaluate_BiHMPgan_motion():
    modelname = "Bihmp-GAN"
    config_id = 0
    epoch = 200
    name = modelname + '_' + str(config_id)
    config = get_config(modelname, config_id)
    ## model initialization
    g = Bihmp_Generator(config["generator"])
    d = Bihmp_Discriminator(config["discriminator"])
    e = Bihmp_Discriminator(config["evaluator"])
    model = BIHMPGAN(g, d, e, name, debug=True)
    model.load(suffix=str(200))

    ## load h36m test data
    input_sequence_length = config['data']['input_sequence_length']
    test_set, mean_pose, std_pose = get_test_data_h36m()
    test_sequence = test_set[:, :input_sequence_length, :]
    test_sequence = (test_sequence - mean_pose) / std_pose 
    ## motion prediction
    model.sample(test_sequence, test_set, std_pose, mean_pose, config['train']["bvh_file"], epoch=epoch)
    ## load 4 motion capture test data
    for dsname in TEST_DS:
        test_data, mean_pose, std_pose = get_test_data_motioncapture(dsname, config['data']['input_sequence_length']+config['data']['output_sequence_length'])
        test_sequence = test_data[:, :input_sequence_length, :]
        test_sequence = (test_sequence - mean_pose) / std_pose
        model.sample(test_sequence, test_data, std_pose, mean_pose, config['train']["bvh_file"], epoch=epoch, sample_path=os.path.join('output', name, 'sample_'+dsname))

def evaluate_BiHMPgan_quality():
    modelname = "Bihmp-GAN"
    config_id = 0
    epoch = 200
    name = modelname + '_' + str(config_id)
    config = get_config(modelname, config_id)
    ## model initialization
    g = Bihmp_Generator(config["generator"])
    d = Bihmp_Discriminator(config["discriminator"])
    e = Bihmp_Discriminator(config["evaluator"])
    model = BIHMPGAN(g, d, e, name, debug=True)
    model.load(suffix=str(200))

    ## load h36m test data
    input_sequence_length = config['data']['input_sequence_length']
    test_set, mean_pose, std_pose = get_test_data_h36m()
    test_set = (test_set - mean_pose) / std_pose 
    ## training data quality
    training_quality = model.e(test_set)
    if type(training_quality) is tuple:
        training_quality = np.mean(training_quality[0].numpy())
    else:
        training_quality = np.mean(training_quality.numpy()) 
    ## load 4 motion capture test data
    test_qualities = []
    for dsname in TEST_DS:
        test_data, mean_pose, std_pose = get_test_data_motioncapture(dsname, config['data']['input_sequence_length']+config['data']['output_sequence_length'])
        test_data = (test_data - mean_pose) / std_pose
        test_quality = model.e(test_data)
        if type(test_quality) is tuple:
            test_quality = np.mean(test_quality[0].numpy())
        else:
            test_quality = np.mean(test_quality.numpy())
        test_qualities.append(test_quality)
    quality = os.path.join('output', name, 'quality.npz')
    np.savez(quality, h36m=training_quality, art=test_qualities[0], captury=test_qualities[1], optitrack=test_qualities[2], vicon=test_qualities[3])

def evaluate_motionGAN_motion():
    modelname = "motionGAN"  
    config_id = 10
    name = modelname + '_' + str(config_id)
    config = get_config(modelname, config_id)
    # name = 'test3'
    # config = load_json_file('C:/Users/Demo/vae_motion_modeling/config/testconfig/config_2.json')
    ## model initialization
    epoch = 200
    g = TCN_RNN_Generator(config["generator"])
    d = CNN_Discriminator(config["discriminator"])
    model = motionGAN_recon(g, d, name, debug=True)
    model.load(suffix=str(200))

    ## load h36m test data
    input_sequence_length = config['data']['input_sequence_length']
    test_set, mean_pose, std_pose = get_test_data_h36m()
    test_sequence = test_set[:, :input_sequence_length, :]
    test_sequence = (test_sequence - mean_pose) / std_pose 
    ## motion prediction
    model.sample(test_sequence, test_set, std_pose, mean_pose, config['train']["bvh_file"], epoch=epoch)
    ## load 4 motion capture test data
    for dsname in TEST_DS:
        test_data, mean_pose, std_pose = get_test_data_motioncapture(dsname, config['data']['input_sequence_length']+config['data']['output_sequence_length'])
        test_sequence = test_data[:, :input_sequence_length, :]
        test_sequence = (test_sequence - mean_pose) / std_pose
        model.sample(test_sequence, test_data, std_pose, mean_pose, config['train']["bvh_file"], epoch=epoch, sample_path=os.path.join('output', name, 'sample_'+dsname))

def evaluate_motionGAN_quality():
    # modelname = "motionGAN"  
    # config_id = 10
    # name = modelname + '_' + str(config_id)
    # config = get_config(modelname, config_id)
    name = 'original_128_nospatial'
    config = load_json_file('config/testconfig/config_nospatial.json')
    ## model initialization
    epoch = 200
    g = TCN_RNN_Generator(config["generator"])
    d = CNN_Discriminator(config["discriminator"])
    model = motionGAN_norecon(g, d, name, debug=True)
    model.load(suffix=str(200))

    ## load h36m test data
    input_sequence_length = config['data']['input_sequence_length']
    test_set, mean_pose, std_pose = get_test_data_h36m()
    test_set = (test_set - mean_pose) / std_pose 
    ## training data quality
    training_quality = model.d(test_set, training=False)
    if type(training_quality) is tuple:
        training_quality = np.mean(training_quality[0].numpy())
    else:
        training_quality = np.mean(training_quality.numpy()) 
    print(training_quality)
    ## load 4 motion capture test data
    test_qualities = []
    for dsname in TEST_DS:
        test_data, mean_pose, std_pose = get_test_data_motioncapture(dsname, config['data']['input_sequence_length']+config['data']['output_sequence_length'])
        test_data = (test_data - mean_pose) / std_pose
        test_quality = model.d(test_data, training=False)
        if type(test_quality) is tuple:
            test_quality = np.mean(test_quality[0].numpy())
        else:
            test_quality = np.mean(test_quality.numpy())
        test_qualities.append(test_quality)
    quality = os.path.join('output', name, 'quality.npz')
    print(test_qualities)
    np.savez(quality, h36m=training_quality, art=test_qualities[0], captury=test_qualities[1], optitrack=test_qualities[2], vicon=test_qualities[3])


def evaluate_spatialencoder():
    name = 'spatialEncoder_new'
    test_set, mean_pose, std_pose = get_test_data_h36m()
    encoder = SpatialEncoder(num_params_per_joint=3, num_zdim=32, joint_dict=CMU_SKELETON, levels=CMU_JOINT_LEVELS)
    encoder.load_weights('output/spatialEncoder_new/model/spatialEncoder_new_SpatialEncoder_300.ckpt')
    data = (test_set - mean_pose) / std_pose
    n_samples, len_seq, dims = data.shape
    x = np.reshape(data, (n_samples*len_seq, -1))
    res = encoder(x)
    print(res.shape)



if __name__ == "__main__":
    evaluate_motionGAN_motion()
    #evaluate_motionGAN_quality()
    # evaluate_spatialencoder()