import os
import sys
import numpy as np 
import tensorflow as tf 
rng = np.random.RandomState(1234567)
mse = tf.keras.losses.MeanSquaredError()


class PoseGAN(object):
    def __init__(self, name, enc, dec, d, z_dims):
        self.name = name
        self.enc = enc
        self.dec = dec
        self.d = d
        self.z_dims = z_dims

    def train(self, training_data, epochs_T1, epochs_T2, batchsize, g_iter=1, d_iter=5, learning_rate=1e-5, save_every_epochs=None, save_path=None, val_data=None, val_every_epochs=None, log_path='./log'):
        '''
        2 phase training:
            phase 1: train the autoencoder with cyclicloss
            phase 2: jointly train the autoencoder and discriminator 
        inputs:
            training_data: motion sequences, [num_frames, num_params_per_frame]
            epochs_T1: number of epochs in phase 1
            epochs_T2: total number of epochs
        '''
        self.enc_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.dec_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.batchsize = batchsize
        n_samples = len(training_data)
        n_batches = n_samples // batchsize
        batch_indexes = np.arange(n_batches)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        train_summary_writer = tf.summary.create_file_writer(log_path)
        for epoch in range(epochs_T2):
            x_recon_errors = []
            z_recon_errors = []
            cyclic_errors = []
            disc_errors = []
            gen_errors = []
            rng.shuffle(batch_indexes)
            for i, batch_index in enumerate(batch_indexes):
                input_batch = training_data[batch_index * batchsize : (batch_index+1) * batchsize]
                if epoch < epochs_T1:
                    x_recon_loss, z_recon_loss, cyclic_loss = self.train_autoencoder(input_batch)
                    x_recon_errors.append(x_recon_loss)
                    z_recon_errors.append(z_recon_loss)
                    cyclic_errors.append(cyclic_loss)
                    with train_summary_writer.as_default():
                        tf.summary.scalar('x_recon_loss', np.mean(x_recon_errors), step=epoch)
                        tf.summary.scalar('z_recon_loss', np.mean(z_recon_errors), step=epoch)
                        tf.summary.scalar('cyclic_loss', np.mean(cyclic_errors), step=epoch)
                    sys.stdout.write('\r[Epoch {epoch}] {percent:.1%} cyclic_loss {cyclic_loss:.5f}'.format(epoch=epoch, percent=i/(n_batches), cyclic_loss=np.mean(cyclic_errors)))        
                    sys.stdout.flush()
                else:
                    for _ in range(d_iter):
                        d_loss = self.train_d(input_batch)
                        disc_errors.append(d_loss)
                        with train_summary_writer.as_default():
                            tf.summary.scalar('disc_loss', np.mean(disc_errors), step=epoch)
                        sys.stdout.write('\r[Epoch {epoch}] {percent:.1%} disc_loss {disc_loss:.5f}'.format(epoch=epoch, percent=i/(n_batches), disc_loss=np.mean(disc_errors)))        
                        sys.stdout.flush()
                    for _ in range(g_iter):
                        cyclic_loss, g_loss = self.train_g(input_batch)
                        cyclic_errors.append(cyclic_loss)
                        gen_errors.append(g_loss)
                        with train_summary_writer.as_default():
                            tf.summary.scalar('cyclic_loss', np.mean(cyclic_errors), step=epoch)
                            tf.summary.scalar('gen_loss', np.mean(gen_errors), step=epoch)
                        sys.stdout.write('\r[Epoch {epoch}] {percent:.1%} gen_loss {gen_loss:.5f}'.format(epoch=epoch, percent=i/(n_batches), gen_loss=np.mean(gen_errors)))        
                        sys.stdout.flush()
            print('')
            if val_data is not None and val_every_epochs is not None:
                if epoch % val_every_epochs == 0:
                    val_x_recon_loss, x_recon = self.test(val_data, batchsize)
                    with train_summary_writer.as_default():
                            tf.summary.scalar('val_x_recon_loss', val_x_recon_loss, step=epoch)
            if save_every_epochs is not None and save_path is not None:
                if (epoch+1) % save_every_epochs == 0:
                    if epoch < epochs_T1:
                        self.save(save_path, epoch+1, 'encoder')
                        self.save(save_path, epoch+1, 'decoder')
                    else:
                        self.save(save_path, epoch+1, 'all')

    @tf.function
    def train_autoencoder(self, x):
        z = tf.random.uniform([self.batchsize, self.z_dims], minval=-1.0, maxval=1.0)
        with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape:
            #x-z-x reconstruction
            encoded = self.enc(x)
            x_recon = self.dec(encoded)
            x_recon_loss = mse(x, x_recon)
            #z-x-z reconstruction
            decoded = self.dec(z)
            z_recon = self.enc(decoded)
            z_recon_loss = mse(z, z_recon)
            cyclic_loss = x_recon_loss + z_recon_loss
        grads_enc = enc_tape.gradient(cyclic_loss, self.enc.trainable_variables)
        grads_dec = dec_tape.gradient(cyclic_loss, self.dec.trainable_variables)
        self.enc_optimizer.apply_gradients(zip(grads_enc, self.enc.trainable_variables))
        self.dec_optimizer.apply_gradients(zip(grads_dec, self.dec.trainable_variables))
        return x_recon_loss, z_recon_loss, cyclic_loss
    
    @tf.function
    def train_d(self,x):
        z = tf.random.uniform([self.batchsize, self.z_dims], minval=-1.0, maxval=1.0)
        with tf.GradientTape() as d_tape:
            fake_input = self.dec(z)
            fake_output = self.d(fake_input)
            real_output = self.d(x)
            disc_loss = self.discriminator_loss(fake_output, fake_input, real_output, x)
        grads_disc = d_tape.gradient(disc_loss, self.d.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(grads_disc, self.d.trainable_variables))
        return disc_loss
    
    def discriminator_loss(self, fake_output, fake_input, real_output, real_input):
        gp_loss = 10.0 * self.gradient_penalty(real_input, fake_input)
        d_loss = tf.reduce_mean(fake_output - real_output) + gp_loss
        return d_loss
    
    def gradient_penalty(self, real, fake):
        alpha = tf.random.uniform([], 0., 1.)
        diff = fake - real
        inter = real + (alpha * diff)
        with tf.GradientTape() as t:
            t.watch(inter)
            pred = self.d(inter)
        grad = t.gradient(pred, [inter])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1]))
        gp = tf.reduce_mean((slopes - 1.)**2)
        return gp
    
    @tf.function
    def train_g(self, x):
        z = tf.random.uniform([self.batchsize, self.z_dims], minval=-1.0, maxval=1.0)
        with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape:
            #x-z-x reconstruction
            encoded = self.enc(x)
            x_recon = self.dec(encoded)
            x_recon_loss = mse(x, x_recon)
            #z-x-z reconstruction
            decoded = self.dec(z)
            z_recon = self.enc(decoded)
            z_recon_loss = mse(z, z_recon)
            cyclic_loss = x_recon_loss + z_recon_loss
            #adv loss
            fake_output = self.d(decoded)
            g_loss = -tf.reduce_mean(fake_output)
            enc_loss = cyclic_loss
            dec_loss = cyclic_loss + 0.02*g_loss
        grads_enc = enc_tape.gradient(enc_loss, self.enc.trainable_variables)
        grads_dec = dec_tape.gradient(dec_loss, self.dec.trainable_variables)
        self.enc_optimizer.apply_gradients(zip(grads_enc, self.enc.trainable_variables))
        self.dec_optimizer.apply_gradients(zip(grads_dec, self.dec.trainable_variables))
        return cyclic_loss, g_loss

    def save(self, save_path, epoch, model):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if model == 'encoder':    
            enc_strings = [self.name, 'SpatialEncoder', str(epoch)]
            self.enc.save_weights(os.path.join(save_path, '_'.join([x for x in enc_strings if x != '']) + '.ckpt'))
        elif model == 'decoder':
            dec_strings = [self.name, 'SpatialDecoder', str(epoch)]
            self.dec.save_weights(os.path.join(save_path, '_'.join([x for x in dec_strings if x != '']) + '.ckpt'))
        elif model == 'all':
            enc_strings = [self.name, 'SpatialEncoder', str(epoch)]
            disc_strings = [self.name, 'SpatialDiscriminator', str(epoch)]
            dec_strings = [self.name, 'SpatialDecoder', str(epoch)]
            self.enc.save_weights(os.path.join(save_path, '_'.join([x for x in enc_strings if x != '']) + '.ckpt'))
            self.dec.save_weights(os.path.join(save_path, '_'.join([x for x in dec_strings if x != '']) + '.ckpt'))
            self.d.save_weights(os.path.join(save_path, '_'.join([x for x in disc_strings if x != '']) + '.ckpt'))
    
    def load(self, save_path, epoch, model):
        if model == 'encoder':    
            enc_strings = [self.name, 'SpatialEncoder', str(epoch)]
            self.enc.load_weights(os.path.join(save_path, '_'.join([x for x in enc_strings if x != '']) + '.ckpt'))
        elif model == 'decoder':
            dec_strings = [self.name, 'SpatialDecoder', str(epoch)]
            self.dec.load_weights(os.path.join(save_path, '_'.join([x for x in dec_strings if x != '']) + '.ckpt'))
        elif model == 'all':
            enc_strings = [self.name, 'SpatialEncoder', str(epoch)]
            disc_strings = [self.name, 'SpatialDiscriminator', str(epoch)]
            dec_strings = [self.name, 'SpatialDecoder', str(epoch)]
            self.enc.load_weights(os.path.join(save_path, '_'.join([x for x in enc_strings if x != '']) + '.ckpt'))
            self.dec.load_weights(os.path.join(save_path, '_'.join([x for x in dec_strings if x != '']) + '.ckpt'))
            self.d.load_weights(os.path.join(save_path, '_'.join([x for x in disc_strings if x != '']) + '.ckpt'))
    
    def test(self, x, batchsize):
        '''
        input:
            x: [batchsize, seq_len, num_params_per_frame] or [batchsize, num_params_per_frame]
        '''
        num_batches = len(x) // batchsize
        x_recon_losses = []
        x_recon_batches = []
        for i in range(num_batches):
            input_batch = x[i*batchsize:(i+1)*batchsize]
            if len(input_batch.shape)==2:
                encoded = self.enc(input_batch)
                x_recon = self.dec(encoded)
                x_recon_loss = mse(input_batch, x_recon)
            elif len(input_batch.shape)==3:
                batch, seq_len, _ = input_batch.shape
                inp = tf.reshape(input_batch, [batch * seq_len, -1])
                encoded = self.enc(inp)
                x_recon = self.dec(encoded)
                x_recon_loss = mse(inp, x_recon)
                x_recon = tf.reshape(x_recon, [batch, seq_len, -1])
            x_recon_losses.append(x_recon_loss.numpy())
            x_recon_batches.append(x_recon)
        recon_loss = np.mean(x_recon_losses)
        recon_batches = tf.concat(x_recon_batches, axis=0)
        return recon_loss, recon_batches



    

        
    







