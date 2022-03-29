import os
import sys
sys.path.append("../..")
from .GAN import GAN
import numpy as np 
import tensorflow as tf 
rng = np.random.RandomState(1234567)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + r'/../..')
from ..utils import smoothness_loss, bone_loss, CMU_SKELETON, smoothness
from mosi_dev_deepmotionmodeling.utilities.utils import get_files, export_point_cloud_data_without_foot_contact, write_to_json_file
from mosi_dev_deepmotionmodeling.mosi_utils_anim.animation_data import BVHReader, SkeletonBuilder, panim

class motionGAN_norecon(GAN):
    def __init__(self, generator, discriminator, name, evaluator=None, debug=False, output_dir='./output'):
        super(motionGAN_norecon, self).__init__(generator, discriminator, evaluator=evaluator, name=name, debug=debug, output_dir=output_dir)
    
    def train(self, input_sequence, output_sequence, test_sequence, test_set, std_pose, mean_pose, config):
        self.g_optimizer = tf.keras.optimizers.Adam(config["learning_rate"]*10)
        self.d_optimizer = tf.keras.optimizers.Adam(config["learning_rate"])
        if self.e is not None:
            self.e_optimizer = tf.keras.optimizers.Adam(config["learning_rate"] / 2)
        self.z_dims = self.g.z_dims
        test_set = test_set * std_pose + mean_pose
        n_samples = len(input_sequence)
        n_batches = n_samples // config["batchsize"]
        batch_indexes = np.arange(n_batches+1)
        # start training
        train_summary_writer = tf.summary.create_file_writer(self.log_path)
        for epoch in range(config["epochs"]):
            disc_errors = []
            gen_errors = []
            g_wgan_errors = []
            d_wgan_errors = []
            if config["content_loss"]:
                self.lambda_c = config["content_loss"]
                content_errors = []
            if config["smooth_loss"]:
                self.lambda_s = config["smooth_loss"]
                smooth_errors = []
            if config["bone_length_loss"]:
                self.lambda_b = config["bone_length_loss"]
                bone_length_errors = []
            if config["guided_loss"]:
                self.lambda_g = config["guided_loss"]
                guided_errors = []
            if self.e is not None:
                eval_errors = []
            rng.shuffle(batch_indexes)
            for i, batch_index in enumerate(batch_indexes):
                if input_sequence[batch_index * config["batchsize"] : (batch_index+1) * config["batchsize"]].size != 0:
                    input_batch = input_sequence[batch_index * config["batchsize"] : (batch_index+1) * config["batchsize"]]
                    output_batch = output_sequence[batch_index * config["batchsize"] : (batch_index+1) * config["batchsize"]]
                    for _ in range(config["d_iter"]):
                        dloss_dict = self.train_d(input_batch, output_batch)
                    for _ in range(config["g_iter"]):
                        gloss_dict = self.train_g(input_batch, output_batch)
                    if self.e is not None and "e_ier" in config:
                        for _ in range(config["e_iter"]):
                            eval_loss = self.train_e(input_batch, output_batch)
                            eval_errors.append(eval_loss.numpy())
                    gen_errors.append(gloss_dict["g_loss"].numpy()) 
                    disc_errors.append(dloss_dict["d_loss"].numpy()) 
                    if self.debug:
                        g_wgan_errors.append(gloss_dict["g_wgan_loss"].numpy())
                        d_wgan_errors.append(dloss_dict["d_wgan_loss"].numpy())
                        with train_summary_writer.as_default():
                            tf.summary.scalar('gen_loss', np.mean(gen_errors), step=epoch)
                            tf.summary.scalar('disc_loss', np.mean(disc_errors), step=epoch)
                            tf.summary.scalar('g_wgan_loss', np.mean(g_wgan_errors), step=epoch)
                            tf.summary.scalar('d_wgan_loss', np.mean(d_wgan_errors), step=epoch)
                            if "content_loss" in gloss_dict:
                                content_errors.append(gloss_dict["content_loss"].numpy())
                                tf.summary.scalar('content_loss', np.mean(content_errors), step=epoch)
                            if "smooth_loss" in gloss_dict:
                                smooth_errors.append(gloss_dict["smooth_loss"].numpy())
                                tf.summary.scalar('smooth_loss', np.mean(smooth_errors), step=epoch)
                            if  "bone_length_loss" in gloss_dict:
                                bone_length_errors.append(gloss_dict["bone_length_loss"].numpy())
                                tf.summary.scalar('bone_loss', np.mean(bone_length_errors), step=epoch)
                            if "guided_loss" in dloss_dict:
                                guided_errors.append(dloss_dict["guided_loss"].numpy())
                                tf.summary.scalar('guided_loss', np.mean(guided_errors), step=epoch)
                            if self.e is not None:
                                tf.summary.scalar('eval_loss', np.mean(eval_errors), step=epoch)
                        sys.stdout.write('\r[Epoch {epoch}] {percent:.1%} gen_loss {gen_loss:.5f} disc_loss {disc_loss:.5f} g_wgan_loss {g_wgan_loss:.5f} d_wgan_loss {d_wgan_loss:.5f}'.format(epoch=epoch, percent=i/(n_batches), 
                                                                                                gen_loss=np.mean(gen_errors), disc_loss=np.mean(disc_errors), g_wgan_loss=np.mean(g_wgan_errors), d_wgan_loss=np.mean(d_wgan_errors)))        
                        sys.stdout.flush()
                    else:
                        with train_summary_writer.as_default():
                            tf.summary.scalar('gen_loss', np.mean(gen_errors), step=epoch)
                            tf.summary.scalar('disc_loss', np.mean(disc_errors), step=epoch)                    
                        sys.stdout.write('\r[Epoch {epoch}] {percent:.1%} gen_loss {gen_loss:.5f} disc_loss {disc_loss:.5f}'.format(epoch=epoch, percent=i/(n_batches), 
                                                                                                    gen_loss=np.mean(gen_errors), disc_loss=np.mean(disc_errors)))        
                        sys.stdout.flush()     
            print('')
            if config["save_every_epochs"] is not None:
                if (epoch+1) % config["save_every_epochs"] == 0:
                    self.save(suffix=str(epoch+1)) 
            if config["sample_every_epochs"] is not None:
                if (epoch+1) % config["sample_every_epochs"] == 0:
                    if self.g.z_dims is not None:
                        self.sample(test_sequence, test_set, std_pose, mean_pose, config["bvh_file"], epoch=epoch+1)
                    else:
                        self.sample(test_sequence, test_set, std_pose, mean_pose, config["bvh_file"], randomness=False, epoch=epoch+1)

    @tf.function
    def train_g(self, input_batch, output_batch):
        with tf.GradientTape() as gen_tape:
            if self.g.z_dims is not None:
                random_noise = tf.random.normal([len(input_batch), self.g.z_dims], dtype=tf.float32)
                prediction = self.g(input_batch, random_noise)
            else:
                prediction = self.g(input_batch)
            fake_input = tf.concat([input_batch, prediction], axis=1)
            real_input = tf.concat([input_batch, output_batch], axis=1)
            fake_output, fake_z, fake_guided = self.d(fake_input)
            real_output, real_z, real_guided = self.d(real_input)
            g_loss, gloss_dict = self.generator_loss(fake_output, input_batch, output_batch, prediction)
        grads_g = gen_tape.gradient(g_loss, self.g.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads_g, self.g.trainable_variables))
        return gloss_dict
        
    def generator_loss(self, fake_output, input_batch, output_batch, prediction):
        gloss_dict = {}
        gloss_dict["g_wgan_loss"] = self.generator_wgan_loss(fake_output)
        g_loss = gloss_dict["g_wgan_loss"]
        if hasattr(self, "lambda_s"):
            data = tf.concat([input_batch[:,-1:,:], prediction], axis=1)
            gloss_dict["smooth_loss"] = smoothness_loss(data)
            g_loss += self.lambda_s * gloss_dict["smooth_loss"]
        if hasattr(self, "lambda_b"):
            ref = input_batch[:, -1, :]
            ref = tf.reshape(ref, [len(input_batch), -1, 3])
            batch, seq_len, _ = prediction.shape
            prediction_res = tf.reshape(prediction, [batch, seq_len, -1, 3])
            gloss_dict["bone_length_loss"] = bone_loss(ref, prediction_res, CMU_SKELETON)
            g_loss += self.lambda_s * gloss_dict["bone_length_loss"]
        if hasattr(self, "lambda_c"):
            gloss_dict["content_loss"] = tf.reduce_mean(tf.keras.losses.MSE(output_batch, prediction))
            g_loss += self.lambda_c * gloss_dict["content_loss"]
        gloss_dict["g_loss"] = g_loss
        return g_loss, gloss_dict
    
    @tf.function
    def train_d(self, input_batch, output_batch):
        with tf.GradientTape() as t:
            if self.g.z_dims is not None:
                random_noise = tf.random.normal([len(input_batch), self.g.z_dims], dtype=tf.float32)
                prediction = self.g(input_batch, random_noise)
            else:
                prediction = self.g(input_batch)
            fake_input = tf.concat([input_batch, prediction], axis=1)
            real_input = tf.concat([input_batch, output_batch], axis=1)
            fake_output, fake_z, fake_guide = self.d(fake_input)
            real_output, real_z, real_guide = self.d(real_input)
            d_loss, dloss_dict = self.discriminator_loss(real_output, real_input, fake_output, fake_input, fake_guide, real_guide)
        grads_d = t.gradient(d_loss, self.d.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads_d, self.d.trainable_variables))
        return dloss_dict
            
    def discriminator_loss(self, real_output, real_input, fake_output, fake_input, fake_guide, real_guide):
        dloss_dict = {}
        dloss_dict["d_wgan_loss"] = self.discriminator_wgan_loss(fake_output, fake_input, real_output, real_input)
        d_loss = dloss_dict["d_wgan_loss"]
        if hasattr(self, "lambda_g"):
            fake_guide_val = smoothness(fake_input)
            real_guide_val = smoothness(real_input)
            dloss_dict["guided_loss"] = tf.reduce_mean(tf.keras.losses.MSE(fake_guide, fake_guide_val)) + tf.reduce_mean(tf.keras.losses.MSE(real_guide, real_guide_val))
            d_loss += self.lambda_g * dloss_dict["guided_loss"]
        dloss_dict["d_loss"] = d_loss
        return d_loss, dloss_dict
    
    @tf.function
    def train_e(self, input_batch, output_batch):
        with tf.GradientTape() as eval_tape:
            if self.g.z_dims is not None:
                random_noise = tf.random.normal([len(input_batch), self.g.z_dims], dtype=tf.float32)
                prediction = self.g(input_batch, random_noise)
            else:
                prediction = self.g(input_batch)
            real_input = tf.concat([input_batch, output_batch], axis=1)
            fake_input = tf.concat([input_batch, predict_batch], axis=1)
            real_output, _, _ = self.e(real_input)
            fake_output, _, _ = self.e(fake_input)
            eval_loss = self.evaluator_loss(real_output, fake_output)
        grads_eval= eval_tape.gradient(eval_loss, self.e.trainable_variables)
        self.e_optimizer.apply_gradients(zip(grads_eval, self.e.trainable_variables))
        return eval_loss
    
