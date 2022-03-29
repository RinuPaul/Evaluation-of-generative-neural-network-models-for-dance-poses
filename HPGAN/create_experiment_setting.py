from mosi_dev_deepmotionmodeling.mosi_utils_anim import write_to_json_file
import os


def create_config_file():
    model_name = "HP-GAN"
    
    config = {}
    config['data'] = {
        "data_path": r'data/training_data/training_clips_h36m.npy',
        "input_sequence_length": 32,
        "output_sequence_length": 32
    }
    config['generator'] = {
        "output_length": 32,
        "output_dims": 96,
        "hidden_units": 1024,
        "z_dims": 128,
        "embed_shape": 128,
        "num_layers": 2,
        "cell": "gru"
    }

    config['discriminator'] = {
        "hidden_size": 512,
        "embed_shape": 128,
        "num_layers": 3,
        "output_activation": None
    }

    config['evaluator'] = {
        "hidden_size": 512,
        "embed_shape": 128,
        "num_layers": 3,
        "output_activation": "sigmoid"
    }

    config['train'] = {
        "epochs": 200,
        "learning_rate": 1e-4,
        "batchsize": 16,
        "save_every_epochs": 5,
        "sample_every_epochs": 5,
        "g_iter": 1, 
        "d_iter": 5, 
        "e_iter": 1,
        "lambda_c": 0.01,
        "lambda_b": 0.1,
        "bvh_file": r'data/training_data/6kmh.bvh'
    }
    if not os.path.exists('./config'):
        os.makedirs('./config')
    save_path = os.path.join("config", model_name + ".json")
    write_to_json_file(save_path, config)


##################################################
    model_name = "Bihmp-GAN"
    
    config = {}
    config['data'] = {
        "data_path": r'data/training_data/training_clips_h36m.npy',
        "input_sequence_length": 32,
        "output_sequence_length": 32
    }
    config['generator'] = {
        "output_length": 32,
        "output_dims": 32,
        "hidden_units": 512,
        "z_dims": 8,
        "cell": "lstm",
        "use_residual": True,
        "enc_file": 'output/spatialEncoder/model/spatialEncoder_SpatialEncoder_150.ckpt',
        "dec_file": 'output/spatialEncoder/model/spatialEncoder_SpatialDecoder_150.ckpt'
    }

    config['discriminator'] = {
        "hidden_size": 512,
        "input_length": 32,
        "output_length": 32,
        "z_dims": config["generator"]["z_dims"],
        "cell": "lstm",
        "bidirectional":True, 
        "use_multi_states":True, 
        "output_activation":None, 
        "enc_file": config["generator"]["enc_file"]
    }

    config['evaluator'] = {
        "hidden_size": 512,
        "input_length": 30,
        "output_length": 30,
        "z_dims": None,
        "cell": "lstm",
        "bidirectional": True, 
        "use_multi_states":True, 
        "output_activation":"sigmoid", 
        "enc_file": config["generator"]["enc_file"]
    }

    config['train'] = {
        "epochs": 200,
        "learning_rate": 1e-4,
        "batchsize": 16,
        "save_every_epochs": 20,
        "sample_every_epochs": 20,
        "g_iter": 1, 
        "d_iter": 5, 
        "e_iter": 1,
        "lambda_c": [0.1, 1],
        "lambda_r": [0.1, 1],
        "bvh_file": r'data/training_data/preprocessed/input_data/preprocessed/Vicon_6kmh.bvh'
    }
    if not os.path.exists('./config'):
        os.makedirs('./config')
    save_path = os.path.join("config", model_name + ".json")
    write_to_json_file(save_path, config)

#################################################
    model_name = "motionGAN"
    
    config = {}
    config['data'] = {
        "data_path": r'data/training_data/training_clips_h36m.npy',
        "input_sequence_length": 32,
        "output_sequence_length": 32
    }


    config['generator'] = {
        "output_length": 32,
        "output_dims": 32,
        "hidden_units": 256,#[128, 256, 512, 1024],
        "z_dims": 32,
        "n_filters": [[128,128,128,128]],
        "f_sizes": [[3,3,3,3]],
        "dilation_rates": [[1,2,4,8]],
        "mid_units": 128,
        "cell": "lstm",
        "use_spatial": True,
        "use_residual": True,
        "enc_file": 'output/spatialEncoder/model/spatialEncoder_SpatialEncoder_150.ckpt',
        "dec_file": 'output/spatialEncoder/model/spatialEncoder_SpatialDecoder_150.ckpt'
    }

    config['discriminator'] = {
        "n_filters": [[128,128,128]],
        "f_sizes": [[3,3,3]],
        "dilation_rates": [[1,2,4]],
        "base_dim": 128,
        "multiscale": [True, False],
        "z_dims": 32,
        "guided": True,
        "output_activation": None
    }

    config['evaluator'] = {
        "n_filters": [[128,128,128]],
        "f_sizes": [[3,3,3]],
        "dilation_rates": [[1,2,4]],
        "base_dim": 128,
        "multiscale": False,
        "z_dims": None,
        "guided": False,
        "output_activation": "sigmoid"
    }

    config['train'] = {
        "epochs": 200,
        "learning_rate": 1e-4,
        "batchsize": 16,
        "save_every_epochs": 20,
        "sample_every_epochs": 20,
        "g_iter": 1, 
        "d_iter": 5, 
        "e_iter": 1,
        "lambda_c": [0.1, 1],
        "lambda_r": [0.1, 1],
        "bvh_file": r'data/training_data/6kmh.bvh',
        "smooth_loss": [None, 0.1],
        "bone_length_loss": [None, 0.1],
        "guided_loss": [None, 0.1]
    }
    if not os.path.exists('./config'):
        os.makedirs('./config')
    save_path = os.path.join("config", model_name + ".json")
    write_to_json_file(save_path, config)


#######################################################
    model_name = "motionGAN_norecon"
    
    config = {}
    config['data'] = {
        "data_path": r'data/training_data/training_clips_h36m.npy',
        "input_sequence_length": 32,
        "output_sequence_length": 32
    }


    config['generator'] = {
        "output_length": 32,
        "output_dims": 32,
        "hidden_units": 256,#[128, 256, 512, 1024],
        "z_dims": 32,
        "n_filters": [[128,128,128,128]],
        "f_sizes": [[3,3,3,3]],
        "dilation_rates": [[1,2,4,4]],
        "mid_units": 128,
        "cell": "lstm",
        "use_spatial": True,
        "use_residual": True,
        "enc_file": 'output/spatialEncoder/model/spatialEncoder_SpatialEncoder_150.ckpt',
        "dec_file": 'output/spatialEncoder/model/spatialEncoder_SpatialDecoder_150.ckpt'
    }

    config['discriminator'] = {
        "n_filters": [[128,128,128]],
        "f_sizes": [[3,3,3]],
        "dilation_rates": [[1,2,4]],
        "base_dim": 128,
        "multiscale": [True, False],
        "z_dims": None,
        "guided": True,
        "output_activation": None
    }

    # config['evaluator'] = {
    #     "n_filters": [[128,128,128]],
    #     "f_sizes": [[3,3,3]],
    #     "dilation_rates": [[1,2,4]],
    #     "base_dim": 128,
    #     "multiscale": True,
    #     "z_dims": None,
    #     "guided": False,
    #     "output_activation": "sigmoid"
    # }

    config['train'] = {
        "epochs": 1,
        "learning_rate": 1e-4,
        "batchsize": 16,
        "save_every_epochs": 20,
        "sample_every_epochs": 20,
        "g_iter": 1, 
        "d_iter": 5, 
        "e_iter": 1,
        "bvh_file": r'data/training_data/preprocessed/input_data/preprocessed/Vicon_6kmh.bvh',
        "smooth_loss": [None, 0.1],
        "bone_length_loss": [None, 0.1],
        "guided_loss": [None, 0.1],
        "content_loss": [None, 0.1]
    }
    if not os.path.exists('./config'):
        os.makedirs('./config')
    save_path = os.path.join("config", model_name + ".json")
    write_to_json_file(save_path, config)



if __name__ == "__main__":
    create_config_file()