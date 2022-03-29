import os
import sys
#sys.path.append('..')
#from GAN import GAN
from .GAN import GAN
import tensorflow as tf 
import numpy as np
import collections
#from ..network.utils import bone_loss, CMU_SKELETON
from ..utils import bone_loss, CMU_SKELETON, smoothness_loss
rng = np.random.RandomState(1234567)
cross_entropy = tf.keras.losses.BinaryCrossentropy()

class HPGAN(GAN):

    def __init__(self, generator, discriminator, evaluator, name, debug=False, output_dir='./output'):
        super(HPGAN, self).__init__(generator, discriminator, evaluator=evaluator, name=name, debug=debug, output_dir=output_dir)
        

        """epochs, batchsize, learning_rate, g_iter=1, d_iter=5, e_iter=1, save_every_epochs=None, lambda_c, lambda_b
        """

    def train(self, input_sequence, output_sequence, test_sequence, test_set, std_pose, mean_pose, config):
        assert len(input_sequence) == len(output_sequence)
        self.g_optimizer = tf.keras.optimizers.Adam(config["learning_rate"])
        self.d_optimizer = tf.keras.optimizers.Adam(config["learning_rate"])
        self.e_optimizer = tf.keras.optimizers.Adam(config["learning_rate"] / 2)
        self.z_dims = self.g.z_dims
        self.lambda_c = config["lambda_c"]
        self.lambda_b = config["lambda_b"]
        test_set = test_set * std_pose + mean_pose
        n_samples = len(input_sequence)
        n_batches = n_samples // config["batchsize"]
        batch_indexes = np.arange(n_batches+1)
        # start training
        train_summary_writer = tf.summary.create_file_writer(self.log_path)
        for epoch in range(config["epochs"]):
            gen_errors = []
            disc_errors = []
            eval_errors = []
            fake_outputs = []
            real_outputs = []
            g_wgan_errors = []
            consistency_errors = []
            bone_length_errors = []
            rng.shuffle(batch_indexes)
            for i, batch_index in enumerate(batch_indexes):
                if input_sequence[batch_index * config["batchsize"] : (batch_index+1) * config["batchsize"]].size != 0:
                    input_batch = input_sequence[batch_index * config["batchsize"] : (batch_index+1) * config["batchsize"]]
                    output_batch = output_sequence[batch_index * config["batchsize"] : (batch_index+1) * config["batchsize"]]
                    for _ in range(config["d_iter"]):
                        disc_loss = self.train_d(input_batch, output_batch)
                    for _ in range(config["e_iter"]):
                        eval_loss = self.train_e(input_batch, output_batch)
                    for _ in range(config["g_iter"]):
                        gen_loss, g_wgan_loss, consistency_loss, bone_length_loss = self.train_g(input_batch, output_batch)
                    gen_errors.append(gen_loss.numpy())
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
                        consistency_errors.append(consistency_loss.numpy())
                        bone_length_errors.append(bone_length_loss.numpy())
                        with train_summary_writer.as_default():
                            tf.summary.scalar('gen_loss', np.mean(gen_errors), step=epoch)
                            tf.summary.scalar('disc_loss', np.mean(disc_errors), step=epoch)
                            tf.summary.scalar('eval_loss', np.mean(eval_errors), step=epoch)
                            tf.summary.scalar('real_likelihood', np.mean(real_outputs), step=epoch)
                            tf.summary.scalar('fake_likelihood', np.mean(fake_outputs), step=epoch)
                            tf.summary.scalar('g_wgan_loss', np.mean(g_wgan_errors), step=epoch)
                            tf.summary.scalar('bone_loss', np.mean(bone_length_errors), step=epoch)
                            tf.summary.scalar('consistency_loss', np.mean(consistency_errors), step=epoch)
                        sys.stdout.write('\r[Epoch {epoch}] {percent:.1%} gen_loss {gen_loss:.5f} disc_loss {disc_loss:.5f} eval_loss {eval_loss:.5f} g_wgan_loss {g_wgan_loss:.5f} bone_loss {bone_loss:.5f} consistency_loss {consistency_loss:.5f}'.format(epoch=epoch, percent=i/(n_batches), 
                                                                                                                                gen_loss=np.mean(gen_errors), disc_loss=np.mean(disc_errors), eval_loss=np.mean(eval_errors), g_wgan_loss=np.mean(g_wgan_errors), bone_loss=np.mean(bone_length_errors), consistency_loss=np.mean(consistency_errors)))        
                        sys.stdout.flush()   
                    else:
                        with train_summary_writer.as_default():
                            tf.summary.scalar('gen_loss', np.mean(gen_errors), step=epoch)
                            tf.summary.scalar('disc_loss', np.mean(disc_errors), step=epoch)   
                            tf.summary.scalar('eval_loss', np.mean(eval_errors), step=epoch)                     
                        sys.stdout.write('\r[Epoch {epoch}] {percent:.1%} gen_loss {gen_loss:.5f} disc_loss {disc_loss:.5f} eval_loss {eval_loss:.5f}'.format(epoch=epoch, percent=i/(n_batches), 
                                                                                                                                gen_loss=np.mean(gen_errors), disc_loss=np.mean(disc_errors), eval_loss=np.mean(eval_errors)))        
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
        with tf.GradientTape() as gen_tape:
            prediction = self.g(input_batch, random_noise)
            fake_input = tf.concat([input_batch, prediction], axis=1)
            fake_output = self.d(fake_input)
            g_loss, g_wgan_loss, consistency_loss, bone_length_loss = self.generator_loss(fake_output, prediction, input_batch)
        grads_g = gen_tape.gradient(g_loss, self.g.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads_g, self.g.trainable_variables))
        return g_loss, g_wgan_loss, consistency_loss, bone_length_loss

    def generator_loss(self, fake_output, prediction, input_batch):
        g_wgan_loss = self.generator_wgan_loss(fake_output)
        data = tf.concat([input_batch[:,-1:,:], prediction], axis=1)
        consistency_loss = smoothness_loss(data)
        ref = input_batch[:, -1, :]
        ref = tf.reshape(ref, [len(input_batch), -1, 3])
        batch, seq_len, _ = prediction.shape
        prediction = tf.reshape(prediction, [batch, seq_len, -1, 3])
        bone_length_loss = bone_loss(ref, prediction, CMU_SKELETON)
        g_loss = g_wgan_loss + self.lambda_c * consistency_loss + self.lambda_b *  bone_length_loss
        return g_loss, g_wgan_loss, consistency_loss, bone_length_loss
    
    @tf.function
    def train_d(self, input_batch, output_batch):
        random_noise = tf.random.normal([len(input_batch), self.g.z_dims], dtype=tf.float32)
        with tf.GradientTape() as disc_tape:
            predict_batch = self.g(input_batch, random_noise)
            real_input = tf.concat([input_batch, output_batch], axis=1)
            fake_input = tf.concat([input_batch, predict_batch], axis=1)
            real_output = self.d(real_input)
            fake_output = self.d(fake_input)
            disc_loss = self.discriminator_loss(fake_output, fake_input, real_output, real_input)
        grads_d = disc_tape.gradient(disc_loss, self.d.trainable_variables)    
        self.d_optimizer.apply_gradients(zip(grads_d, self.d.trainable_variables))
        return disc_loss 

    def discriminator_loss(self, fake_output, fake_input, real_output, real_input):
        return self.discriminator_wgan_loss(fake_output, fake_input, real_output, real_input) + sum(self.d.losses)
    
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

    def evaluator_loss(self, real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        eval_loss = real_loss + fake_loss + sum(self.e.losses)
        return eval_loss