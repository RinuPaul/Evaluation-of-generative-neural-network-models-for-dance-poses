import os
import sys
sys.path.append("../..")
from .GAN import GAN
import tensorflow as tf 
import numpy as np
import collections
from ..utils import bone_loss, CMU_SKELETON, smoothness_loss
from mosi_dev_deepmotionmodeling.utilities.utils import get_files, export_point_cloud_data_without_foot_contact, write_to_json_file
from mosi_dev_deepmotionmodeling.mosi_utils_anim.animation_data import BVHReader, SkeletonBuilder, panim
rng = np.random.RandomState(1234567)
cross_entropy = tf.keras.losses.BinaryCrossentropy()


class BIHMPGAN(GAN):
    def __init__(self, generator, discriminator, evaluator, name, debug=False, output_dir='./output'):
        super(BIHMPGAN, self).__init__(generator, discriminator, evaluator=evaluator, name=name, debug=debug, output_dir=output_dir)
        
        '''epochs, batchsize, lambda_r=1, lambda_c=0.1, d_iterations=5, g_iterations=1, sample_batchsize=10,
         sample_every_epochs=None, save_every_epochs=None,'''

    def train(self, input_sequence, output_sequence, test_sequence, test_set, std_pose, mean_pose, config):
        self.enc_optimizer = tf.keras.optimizers.Adam(config["learning_rate"])
        self.dec_optimizer = tf.keras.optimizers.Adam(config["learning_rate"])
        self.d_optimizer = tf.keras.optimizers.Adam(config["learning_rate"])
        self.e_optimizer = tf.keras.optimizers.Adam(config["learning_rate"] / 2)
        self.z_dims = self.g.z_dims
        self.lambda_c = config["lambda_c"]
        self.lambda_r = config["lambda_r"]
        test_set = test_set * std_pose + mean_pose
        self.sample(test_sequence, test_set, std_pose, mean_pose, config["bvh_file"], build=True)
        n_samples = len(input_sequence)
        n_batches = n_samples // config["batchsize"]
        batch_indexes = np.arange(n_batches+1)
        # start training
        train_summary_writer = tf.summary.create_file_writer(self.log_path)
        for epoch in range(config["epochs"]):
            disc_errors = []
            enc_errors = []
            dec_errors =[]
            eval_errors = []
            g_wgan_errors = []
            d_wgan_errors = []
            r_recon_errors = []
            content_errors = []
            fake_outputs = []
            real_outputs = []
            rng.shuffle(batch_indexes)
            for i, batch_index in enumerate(batch_indexes):
                if input_sequence[batch_index * config["batchsize"] : (batch_index+1) * config["batchsize"]].size != 0:
                    input_batch = input_sequence[batch_index * config["batchsize"] : (batch_index+1) * config["batchsize"]]
                    output_batch = output_sequence[batch_index * config["batchsize"] : (batch_index+1) * config["batchsize"]]
                    for _ in range(config["d_iter"]):
                        disc_loss, d_wgan_loss, r_recon_loss = self.train_d(input_batch, output_batch)
                    for _ in range(config["e_iter"]):
                        eval_loss = self.train_e(input_batch, output_batch)
                    for _ in range(config["g_iter"]):
                        enc_loss, dec_loss, g_wgan_loss, content_loss, r_recon_loss = self.train_g(input_batch, output_batch)
                    enc_errors.append(enc_loss.numpy())
                    dec_errors.append(dec_loss.numpy())
                    disc_errors.append(disc_loss.numpy()) 
                    eval_errors.append(eval_loss.numpy())
                    if self.debug:
                        random_noise = tf.random.normal([len(input_batch), self.g.z_dims], dtype=tf.float32)
                        predict_batch = self.g(input_batch, random_noise)
                        real_input = tf.concat([input_batch, output_batch], axis=1)
                        fake_input = tf.concat([input_batch, predict_batch], axis=1)
                        real_prob = self.e(real_input)
                        fake_prob = self.e(fake_input)
                        fake_outputs.append(np.mean(fake_prob.numpy()))
                        real_outputs.append(np.mean(real_prob.numpy()))
                        g_wgan_errors.append(g_wgan_loss.numpy())
                        d_wgan_errors.append(d_wgan_loss.numpy())
                        r_recon_errors.append(r_recon_loss.numpy())
                        content_errors.append(content_loss.numpy())
                        with train_summary_writer.as_default():
                            tf.summary.scalar('enc_loss', np.mean(enc_errors), step=epoch)
                            tf.summary.scalar('dec_loss', np.mean(dec_errors), step=epoch)
                            tf.summary.scalar('disc_loss', np.mean(disc_errors), step=epoch)
                            tf.summary.scalar('eval_loss', np.mean(eval_errors), step=epoch)
                            tf.summary.scalar('real_likelihood', np.mean(real_outputs), step=epoch)
                            tf.summary.scalar('fake_likelihood', np.mean(fake_outputs), step=epoch)
                            tf.summary.scalar('g_wgan_loss', np.mean(g_wgan_errors), step=epoch)
                            tf.summary.scalar('d_wgan_loss', np.mean(d_wgan_errors), step=epoch)
                            tf.summary.scalar('r_recon_loss', np.mean(r_recon_errors), step=epoch)
                            tf.summary.scalar('content_loss', np.mean(content_errors), step=epoch)
                        sys.stdout.write('\r[Epoch {epoch}] {percent:.1%} enc_loss {enc_loss:.5f} dec_loss {dec_loss:.5f} disc_loss {disc_loss:.5f} eval_loss {eval_loss:.5f} g_wgan_loss {g_wgan_loss:.5f} d_wgan_loss {d_wgan_loss:.5f} r_recon_loss {r_recon_loss:.5f} content_loss {content_loss:.5f}'.format(epoch=epoch, percent=i/(n_batches), 
                                                                                                enc_loss=np.mean(enc_errors), dec_loss=np.mean(dec_errors), disc_loss=np.mean(disc_errors), eval_loss=np.mean(eval_errors), g_wgan_loss=np.mean(g_wgan_errors), d_wgan_loss=np.mean(d_wgan_errors), r_recon_loss=np.mean(r_recon_errors), content_loss=np.mean(content_errors)))        
                        sys.stdout.flush()
                    else:
                        with train_summary_writer.as_default():
                            tf.summary.scalar('enc_loss', np.mean(enc_errors), step=epoch)
                            tf.summary.scalar('dec_loss', np.mean(dec_errors), step=epoch)
                            tf.summary.scalar('disc_loss', np.mean(disc_errors), step=epoch)   
                            tf.summary.scalar('eval_loss', np.mean(eval_errors), step=epoch)                     
                        sys.stdout.write('\r[Epoch {epoch}] {percent:.1%} enc_loss {enc_loss:.5f} dec_loss {dec_loss:.5f} disc_loss {disc_loss:.5f} eval_loss {eval_loss:.5f}'.format(epoch=epoch, percent=i/(n_batches), 
                                                                                                    enc_loss=np.mean(enc_errors), dec_loss=np.mean(dec_errors), disc_loss=np.mean(disc_errors), eval_loss=np.mean(eval_errors)))        
                        sys.stdout.flush()     
            print('')
            if config["save_every_epochs"] is not None:
                if (epoch+1) % config["save_every_epochs"] == 0:
                    self.save(suffix=str(epoch+1)) 
            if config["sample_every_epochs"] is not None:
                if (epoch+1) % config["sample_every_epochs"] == 0:
                    self.sample(test_sequence, test_set, std_pose, mean_pose, config["bvh_file"], epoch=epoch+1)

    @tf.function
    def train_g(self, input_batch, output_batch):
        random_noise = tf.random.normal([len(input_batch), self.g.z_dims], dtype=tf.float32)
        with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape:
            prediction = self.g(input_batch, random_noise)
            fake_input = tf.concat([input_batch, prediction], axis=1)
            real_input = tf.concat([input_batch, output_batch], axis=1)
            fake_output, fake_z = self.d(fake_input)
            real_output, real_z = self.d(real_input)
            content = self.g(input_batch, real_z)
            enc_loss, dec_loss, g_wgan_loss, content_loss, r_recon_loss = self.generator_loss(fake_output, random_noise, fake_z, content, output_batch)
        grads_enc = enc_tape.gradient(enc_loss, self.enc_trainable_variables)
        grads_dec = dec_tape.gradient(dec_loss, self.dec_trainable_variables)
        self.enc_optimizer.apply_gradients(zip(grads_enc, self.enc_trainable_variables))
        self.dec_optimizer.apply_gradients(zip(grads_dec, self.dec_trainable_variables))
        return enc_loss, dec_loss, g_wgan_loss, content_loss, r_recon_loss
        
    def generator_loss(self, fake_output, random_noise, recon_z, content, output_batch):
        g_wgan_loss = self.generator_wgan_loss(fake_output)
        r_recon_loss = tf.maximum(0.0001, tf.reduce_mean(tf.norm(random_noise-recon_z,axis=-1)))
        content_loss = tf.maximum(0.0001, tf.reduce_mean(tf.norm(output_batch-content, axis=-1)))
        enc_loss = g_wgan_loss + self.lambda_c * content_loss
        dec_loss = g_wgan_loss + self.lambda_r * r_recon_loss + self.lambda_c * content_loss
        return enc_loss, dec_loss, g_wgan_loss, content_loss, r_recon_loss
    
    @tf.function
    def train_d(self, input_batch, output_batch):
        random_noise = tf.random.normal([len(input_batch), self.g.z_dims], dtype=tf.float32)
        with tf.GradientTape() as t:
            prediction = self.g(input_batch, random_noise)
            fake_input = tf.concat([input_batch, prediction], axis=1)
            real_input = tf.concat([input_batch, output_batch], axis=1)
            fake_output, fake_z = self.d(fake_input)
            real_output, real_z = self.d(real_input)
            d_loss, d_wgan_loss, r_recon_loss = self.discriminator_loss(real_output, real_input, fake_output, fake_input, random_noise, fake_z)
        grads_d = t.gradient(d_loss, self.d.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads_d, self.d.trainable_variables))
        return d_loss, d_wgan_loss, r_recon_loss
            
    def discriminator_loss(self, real_output, real_input, fake_output, fake_input, random_noise, recon_z):
        d_wgan_loss = self.discriminator_wgan_loss(fake_output, fake_input, real_output, real_input)
        r_recon_loss = tf.maximum(0.0001, tf.reduce_mean(tf.norm(random_noise-recon_z, axis=-1)))
        d_loss = d_wgan_loss + self.lambda_r * r_recon_loss
        return d_loss, d_wgan_loss, r_recon_loss
    
    @tf.function
    def train_e(self, input_batch, output_batch):
        random_noise = tf.random.normal([len(input_batch), self.z_dims], dtype=tf.float32)
        with tf.GradientTape() as eval_tape:
            predict_batch = self.g(input_batch, random_noise)
            real_input = tf.concat([input_batch, output_batch], axis=1)
            fake_input = tf.concat([input_batch, predict_batch], axis=1)
            real_output = self.e(real_input)
            fake_output = self.e(fake_input)
            eval_loss = self.evaluator_loss(real_output, fake_output)
        grads_eval= eval_tape.gradient(eval_loss, self.e.trainable_variables)
        self.e_optimizer.apply_gradients(zip(grads_eval, self.e.trainable_variables))
        return eval_loss

    def sample(self, input_batch, ground_truth, std_pose, mean_pose, bvh_file, randomness=True, build=False, epoch=0, sample_path=None):
        if sample_path is None:
            sample_path = self.sample_path
        sample_dir = os.path.join(sample_path, str(epoch))
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        if randomness:
            random_noise = tf.random.normal([len(input_batch), self.g.z_dims], dtype=tf.float32)
            prediction = self.g(input_batch, random_noise)
        else:
            prediction = self.g(input_batch)
        generated_sequence = tf.concat([input_batch, prediction], axis=1)
        generated_sequence = generated_sequence * std_pose + mean_pose  
        skeleton_bvh = BVHReader(bvh_file)
        skeleton = SkeletonBuilder().load_from_bvh(skeleton_bvh)
        animated_joints = skeleton.generate_bone_list_description() 
        if isinstance(animated_joints, list):
            animated_joints = collections.OrderedDict([(d['name'], {'parent':d['parent'], 'index':d['index']}) for d in animated_joints])
        for i in range(len(generated_sequence)):
            filename = os.path.join(sample_dir, 'gen_' + str(i) + '.panim')
            export_point_cloud_data_without_foot_contact(generated_sequence[i].numpy(), filename, skeleton=animated_joints)
            truth_filename = os.path.join(sample_dir, 'truth_' + str(i) + '.panim')
            export_point_cloud_data_without_foot_contact(ground_truth[i], truth_filename, skeleton=animated_joints)
        if build:
            self.enc_trainable_variables = []
            self.dec_trainable_variables = []
            for v in self.g.trainable_variables:
                if 'encoder' in v.name:
                    self.enc_trainable_variables.append(v)
                if 'decoder' in v.name:
                    self.dec_trainable_variables.append(v)

