""" 
    Class for Aligned Autoencoder

    >>> python3 autoencoder.py [pickle_path]
"""

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
import sys
from autoencoder_utils import *

class Aligned_LSTM_Autoencoder:
    def __init__(self, learning_rate=0.0001, temperature=0.001, beta1=0.9, beta2=0.9, z_dim=100, lambda_g=1):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.z_dim = z_dim
        self.lambda_g = lambda_g
        self.temperature = temperature

    def graph_init(self, vocab_size, max_length):
        print("Initializing computation graph...")
        tf.reset_default_graph()

        # Build Autoencoder1
        self.x1_input = tf.placeholder(tf.int32, [None, None])
        embedding1 = tf.one_hot(self.x1_input, vocab_size)

        self.encoder_output1 = lstm_encoder(embedding1, lstm_units=vocab_size, z_dim=self.z_dim, name='1')
        self.decoder_output1 = lstm_decoder_teacher_forced(embedding1, self.encoder_output1, max_length, lstm_units=vocab_size, name='1')

        # Build Autoencoder2
        self.x2_input = tf.placeholder(tf.int32, [None, None])
        embedding2 = tf.one_hot(self.x2_input, vocab_size)

        self.encoder_output2 = lstm_encoder(embedding2, lstm_units=vocab_size, z_dim=self.z_dim, name='2')
        self.decoder_output2 = lstm_decoder_teacher_forced(embedding2, self.encoder_output2, max_length, lstm_units=vocab_size, \
                                                        training=False, softmax=True, temperature=self.temperature, name='2')

        # Reconstruction Loss
        self.rec_loss1 = tf.reduce_mean(tf.square(self.decoder_output1 - embedding1))
        self.rec_loss2 = tf.reduce_mean(tf.square(self.decoder_output2 - embedding2))

        # Define Discrimator
        with tf.variable_scope("adversary") as scope:
            logits1 = discriminator(self.encoder_output1)
            logits2 = discriminator(self.encoder_output2, reuse=True)

        # Adversarial Loss
        _, adv_loss = gan_loss(logits1, logits2)
        
        # Define losses
        self.G_loss = (self.rec_loss1 + self.rec_loss2) + self.lambda_g * (adv_loss)
        self.D_loss = adv_loss

        #Define optimizers
        D_solver, G_solver = get_solvers()
        self.G_train_step = G_solver.minimize(self.G_loss)
        self.D_train_step = D_solver.minimize(self.D_loss)

        # Write summaries
        tf.summary.scalar('rec_loss1', self.rec_loss1)
        tf.summary.scalar('rec_loss2', self.rec_loss2)
        tf.summary.scalar('G_loss', self.G_loss)
        tf.summary.scalar('D_loss', self.D_loss)
        self.merged = tf.summary.merge_all()

    def load_data(self, pickle_path):
        print("Loading data...")
        with open(pickle_path, "rb") as f:
            pickle_dict = pickle.load(f)

            self.X1 = pickle_dict['X1']
            self.X2 = pickle_dict['X2']

            classes = pickle_dict['classes']
            self.encoder = LabelEncoder()
            self.encoder.classes_ = classes
            
            self.vocab_size = len(classes)
            self.max_length = self.X1.shape[1]
            print("Vocab size: {0}".format(self.vocab_size))
            print("Max Length: {0}".format(self.max_length))

    def train(self, pickle_path, num_epochs=10, batch_size=100, \
               saved_model_path='../models/aligned_lstm_ae/aligned_lstm_ae.ckpt', \
               checkpoint_dir='../models/aligned_lstm_ae/'):
        # Get X1 and X2 from pickle_path
        self.load_data(pickle_path)

        # Define computation graph
        self.graph_init(self.vocab_size, self.max_length)

        # Train
        print("Starting Training...")
        init = tf.global_variables_initializer()
	
        saver = tf.train.Saver()
        train_writer = tf.summary.FileWriter(checkpoint_dir + 'train')

        step = 0
        with tf.Session() as sess:
            # Load model
            try:
      	        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
            except:
                print('No model')
                sess.run(init)

            # Start Training
            for i in range(num_epochs):
                num_batches = int(self.X1.shape[0] / batch_size)
                for b in range(num_batches):
                    x1_batch = self.X1[b * batch_size:(b + 1) * batch_size]
                    x2_batch = self.X2[b * batch_size:(b + 1) * batch_size]

                    _, _, summary = sess.run([self.D_train_step, self.G_train_step, self.merged], \
                                feed_dict={self.x1_input: x1_batch, self.x2_input: x2_batch})
                    
                    if i % 1 == 0 and b == 0:
                        D_loss, G_loss = sess.run([self.D_loss, self.G_loss], \
                                feed_dict={self.x1_input: x1_batch, self.x2_input: x2_batch})
                        rec_loss1, rec_loss2 = sess.run([self.rec_loss1, self.rec_loss2], \
                                feed_dict={self.x1_input: x1_batch, self.x2_input: x2_batch})
                        print("D_loss: {0}\n G_loss: {1}".format(D_loss, G_loss))
                        print("rec_loss1: {0}\n rec_loss2: {1}".format(rec_loss1, rec_loss2))

			
                    x_gen_i, x_transfered_i = sess.run([self.decoder_output1, self.decoder_output2], \
                            feed_dict={self.x1_input: x1_batch, self.x2_input: x2_batch})
                    x_gen_i = np.argmax(x_gen_i, axis=2)
                    x_transferred_i = np.argmax(x_transferred_i, axis=2)
                    
                    print(i)
                    num_show = 5
                    show_idx = np.random.choice(x1_batch.shape[0], size=num_show)
                    print(textify_samples(x1_batch[show_idx]))
                    print(textify_samples(x_gen_i[show_idx]))
                    print(textify_samples(x2_batch[show_idx]))
                    print(textify_samples(x_transferred_i[show_idx]))
    
                    saver.save(sess, saved_model_path, global_step=step)

                    step += batch_size

######################
### Main Interface ###
######################
def main(argv):
    pickle_path = argv[0]
    aae = Aligned_LSTM_Autoencoder()
    aae.train(pickle_path)

if __name__ == '__main__':
    main(sys.argv[1:])
