import os
import shutil
import numpy as np
from Config import Config
from mosi_dev_deepmotionmodeling.mosi_utils_anim import load_json_file, write_to_json_file
from mosi_dev_deepmotionmodeling.preprocessing.preprocessing import process_file
from src_final.network.Discriminator import HPDiscriminator, Bihmp_Discriminator, CNN_Discriminator, CNN_Discriminator_test1, CNN_Discriminator_test2, CNN_Discriminator_test3
from src_final.network.Generator import HPGenerator,Bihmp_Generator, TCN_RNN_Generator
from src_final.model.HPGAN import HPGAN
from src_final.model.BIHMPGAN import BIHMPGAN
from src_final.model.motionGAN_recon import motionGAN_recon
from src_final.model.motionGAN_norecon import motionGAN_norecon
from src_final.utils import CMU_SKELETON, CMU_ANIMATED_JOINTS
import itertools
import copy
EPS = 1e-6


def create_configs(name, save=True):
    """create a list of configuration

    Args:
        name (str): test model name

    """
    save_dir = os.path.join("config", name)
    if os.path.exists(save_dir):
        files = [os.path.join(save_dir, f) for f in os.listdir(save_dir)]
        configs = []
        for f in files:
            config = load_json_file(f)
            configs.append((config, f))
        return configs
    if not os.path.exists(os.path.join("config", name+".json")):
        raise ValueError("Cannot find configure setting, please run create_experiemnt_setting.py to create configuration")
    else:
        config_paramters = load_json_file(os.path.join("config", name+".json"))
    configs = []
    ### loop over all the parameters and find the parameter with more that one value
    param_labels = []
    param_lists = []
    for config_type in config_paramters.keys():
        for param, value in config_paramters[config_type].items():
            if type(value) is list:
                param_labels.append('__'.join([config_type, param]))
                param_lists.append(value)
    combined = list(itertools.product(*param_lists))
    print("{} parameter setting is created.".format(len(combined)))
    for i in range(len(combined)):
        new_config = copy.deepcopy(config_paramters)
        assert len(param_labels) == len(combined[i])
        for j in range(len(param_labels)):
            config_type, param = param_labels[j].split("__")
            try:
                new_config[config_type][param] = combined[i][j]
            except:
                print("Cannot find parameter {} in the configuration.".format(param))
        configs.append(new_config)
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for id in range(len(configs)):
            write_to_json_file(os.path.join(save_dir, "config_{}.json".format(id)), configs[id])
            configs[id] = (configs[id], os.path.join(save_dir, "config_{}.json".format(id)))
    return configs

def main():
    
    ### experiment setting
    model_to_be_trained = ["motionGAN_norecon"]

    ### HP-GAN
    if "HP-GAN" in model_to_be_trained:
        modelname = "HP-GAN"
        configs = create_configs(modelname)
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
        print("{} configurations will be trained".format(len(configs))) 
        for config, filename in configs:
            basename = os.path.basename(filename)
            i = basename.split('.')[0].split('_')[1]
            name = modelname + '_' + str(i)
            output_dir = os.path.join('output', name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            ## load training data
            training_data = np.load(config['data']['data_path'])
            training_data = np.asarray(training_data, dtype=np.float32)
            mean_pose = training_data.mean(axis=(0, 1))
            std_pose = training_data.std(axis=(0, 1))
            std_pose[std_pose<EPS] = EPS
            training_data = (training_data - mean_pose) / std_pose   
            n_samples, sequence_length, dims = training_data.shape
            train_set = training_data[:int(0.9*n_samples)] 
            test_set = training_data[int(0.9*n_samples):]      

            input_sequence_length = config['data']['input_sequence_length']
            output_sequence_length = config['data']['output_sequence_length']
            input_sequence = train_set[:, :input_sequence_length, :]
            output_sequence = train_set[:, input_sequence_length:, :]    
            test_sequence = test_set[:, :input_sequence_length, :]        
           

            ## model initialization
            g = HPGenerator(config["generator"])
            d = HPDiscriminator(config["discriminator"])
            e = HPDiscriminator(config["evaluator"])
            model = HPGAN(g, d, e, name, debug=True, output_dir=output_dir)
            model.train(input_sequence, output_sequence, test_sequence, test_set, std_pose, mean_pose, config["train"])
            shutil.move(filename, output_dir)

    ### Bihmp-GAN
    if "Bihmp-GAN" in model_to_be_trained:
        modelname = "Bihmp-GAN"
        configs = create_configs(modelname)
        print("{} configurations will be trained".format(len(configs))) 
        for config, filename in configs:
            basename = os.path.basename(filename)
            i = basename.split('.')[0].split('_')[1]
            name = modelname + '_' + str(i)
            output_dir = os.path.join('output', name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            ## load training data
            training_data = np.load(config['data']['data_path'])
            training_data = np.asarray(training_data, dtype=np.float32)
            mean_pose = training_data.mean(axis=(0, 1))
            std_pose = training_data.std(axis=(0, 1))
            std_pose[std_pose<EPS] = EPS
            training_data = (training_data - mean_pose) / std_pose   
            n_samples, sequence_length, dims = training_data.shape
            train_set = training_data[:int(0.9*n_samples)] 
            test_set = training_data[int(0.9*n_samples):]      

            input_sequence_length = config['data']['input_sequence_length']
            output_sequence_length = config['data']['output_sequence_length']
            input_sequence = train_set[:, :input_sequence_length, :]
            output_sequence = train_set[:, input_sequence_length:, :]    
            test_sequence = test_set[:, :input_sequence_length, :]        


            ## model initialization
            g = Bihmp_Generator(config["generator"])
            d = Bihmp_Discriminator(config["discriminator"])
            e = Bihmp_Discriminator(config["evaluator"])
            model = BIHMPGAN(g, d, e, name, debug=True, output_dir=output_dir)
            model.train(input_sequence, output_sequence, test_sequence, test_set, std_pose, mean_pose, config["train"])
            shutil.move(filename, output_dir)
            
    ###motionGAN
    if "motionGAN" in model_to_be_trained:
        modelname = "motionGAN"
        configs = create_configs(modelname)
        print("{} configurations will be trained".format(len(configs))) 
        #for i, config in enumerate(configs):
        for config, filename in configs:
            basename = os.path.basename(filename)
            i = basename.split('.')[0].split('_')[1]
            name = modelname + '_' + str(i)
            output_dir = os.path.join('output', name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            ## load training data
            training_data = np.load(config['data']['data_path'])
            training_data = np.asarray(training_data, dtype=np.float32)
            mean_pose = training_data.mean(axis=(0, 1))
            std_pose = training_data.std(axis=(0, 1))
            std_pose[std_pose<EPS] = EPS
            training_data = (training_data - mean_pose) / std_pose   
            n_samples, sequence_length, dims = training_data.shape
            train_set = training_data[:int(0.9*n_samples)] 
            test_set = training_data[int(0.9*n_samples):]      

            input_sequence_length = config['data']['input_sequence_length']
            output_sequence_length = config['data']['output_sequence_length']
            input_sequence = train_set[:, :input_sequence_length, :]
            output_sequence = train_set[:, input_sequence_length:, :]    
            test_sequence = test_set[:, :input_sequence_length, :]        


            ## model initialization
            g = TCN_RNN_Generator(config["generator"])
            d = CNN_Discriminator(config["discriminator"])
            model = motionGAN_recon(g, d, name, debug=True, output_dir=output_dir)
            model.train(input_sequence, output_sequence, test_sequence, test_set, std_pose, mean_pose, config["train"])
            shutil.move(filename, output_dir)

    ###motionGAN
    if "motionGAN_norecon" in model_to_be_trained:
        modelname = "motionGAN_norecon"
        configs = create_configs(modelname)
        print("{} configurations will be trained".format(len(configs))) 
        for config, filename in configs:
            print(filename)
            basename = os.path.basename(filename)
            i = basename.split('.')[0].split('_')[1]
            name = modelname + '_' + str(i)
            output_dir = os.path.join('output', name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            ## load training data
            training_data = np.load(config['data']['data_path'])
            training_data = np.asarray(training_data, dtype=np.float32)
            mean_pose = training_data.mean(axis=(0, 1))
            std_pose = training_data.std(axis=(0, 1))
            std_pose[std_pose<EPS] = EPS
            training_data = (training_data - mean_pose) / std_pose   
            n_samples, sequence_length, dims = training_data.shape
            train_set = training_data[:int(0.9*n_samples)] 
            test_set = training_data[int(0.9*n_samples):]      

            input_sequence_length = config['data']['input_sequence_length']
            output_sequence_length = config['data']['output_sequence_length']
            input_sequence = train_set[:, :input_sequence_length, :]
            output_sequence = train_set[:, input_sequence_length:, :]    
            test_sequence = test_set[:, :input_sequence_length, :]        


            ## model initialization
            g = TCN_RNN_Generator(config["generator"])
            d = CNN_Discriminator(config["discriminator"])
            model = motionGAN_norecon(g, d, name, debug=True, output_dir=output_dir)
            model.train(input_sequence, output_sequence, test_sequence, test_set, std_pose, mean_pose, config["train"])
            shutil.move(filename, output_dir)



            
    

if __name__ == "__main__":
    main()