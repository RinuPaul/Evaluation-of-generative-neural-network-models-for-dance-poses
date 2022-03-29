import os
import tensorflow as tf 
import sys
sys.path.append("../..")
import collections
from mosi_dev_deepmotionmodeling.utilities.utils import get_files, export_point_cloud_data_without_foot_contact, write_to_json_file
from mosi_dev_deepmotionmodeling.mosi_utils_anim.animation_data import BVHReader, SkeletonBuilder, panim
cross_entropy = tf.keras.losses.BinaryCrossentropy()

class GAN(object):

    def __init__(self, generator, discriminator, evaluator=None, name=None, debug=False, output_dir='./output'):
        self.g = generator
        self.d = discriminator
        self.e = evaluator
        self.name = name
        self.debug = debug
        self.init_paths(output_dir)
    
    def init_paths(self, output_dir):
        if output_dir == './output':
            output_dir = os.path.join(output_dir, self.name)
        self.log_path = os.path.join(output_dir, 'log')
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.save_path = os.path.join(output_dir, 'model')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.sample_path = os.path.join(output_dir, 'sample')
        if not os.path.exists(self.sample_path):
            os.makedirs(self.sample_path)

    def generator_wgan_loss(self, fake_output):
        return  -tf.reduce_mean(fake_output)
        
    def discriminator_wgan_loss(self, fake_output, fake_input, real_output, real_input):
        d_loss = tf.reduce_mean(fake_output - real_output) + 10.0 * self.gradient_penalty(real_input, fake_input)
        return d_loss

    def gradient_penalty(self, real, fake):
        alpha = tf.random.uniform([], 0., 1.)
        diff = fake - real
        inter = real + (alpha * diff)
        with tf.GradientTape() as t:
            t.watch(inter)
            pred = self.d(inter)
        if type(pred) is tuple:
            grad = t.gradient(pred[0], [inter])[0]
        else:
            grad = t.gradient(pred, [inter])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1,2]))
        gp = tf.reduce_mean((slopes - 1.)**2)
        return gp
    
    def evaluator_loss(self, real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        eval_loss = real_loss + fake_loss
        return eval_loss

    def save(self, prefix='', suffix=''):
        if self.g is not None:
            gen_strings = [self.name, prefix, 'generator', suffix]
            self.g.save_weights(os.path.join(self.save_path, '_'.join([x for x in gen_strings if x != '']) + '.ckpt'))
        if self.d is not None:
            disc_strings = [self.name, prefix, 'discriminator', suffix]
            self.d.save_weights(os.path.join(self.save_path, '_'.join([x for x in disc_strings if x != '']) + '.ckpt')) 
        if self.e is not None:
            eval_strings = [self.name, prefix, 'evaluator', suffix]
            self.e.save_weights(os.path.join(self.save_path, '_'.join([x for x in eval_strings if x != '']) + '.ckpt')) 

    def load(self, prefix='', suffix=''):
        if self.g is not None:
            gen_strings = [self.name, prefix, 'generator', suffix]
            self.g.load_weights(os.path.join(self.save_path, '_'.join([x for x in gen_strings if x != '']) + '.ckpt'))
        if self.d is not None:
            disc_strings = [self.name, prefix, 'discriminator', suffix]
            self.d.load_weights(os.path.join(self.save_path, '_'.join([x for x in disc_strings if x != '']) + '.ckpt')) 
        if self.e is not None:
            eval_strings = [self.name, prefix, 'evaluator', suffix]
            self.e.load_weights(os.path.join(self.save_path, '_'.join([x for x in eval_strings if x != '']) + '.ckpt')) 
        
    def sample(self, input_batch, ground_truth, std_pose, mean_pose, bvh_file, randomness=True, epoch=0, sample_path=None):
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
            filename = os.path.join(sample_dir, 'gen_' + str(epoch) + '_' + str(i) + '.panim')
            export_point_cloud_data_without_foot_contact(generated_sequence[i].numpy(), filename, skeleton=animated_joints)
            truth_filename = os.path.join(sample_dir, 'truth_' + str(epoch) + '_' + str(i) + '.panim')
            export_point_cloud_data_without_foot_contact(ground_truth[i], truth_filename, skeleton=animated_joints)
    

    
    