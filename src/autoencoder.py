import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle

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
        x1_input = tf.placeholder(tf.int32, [None, None])
        embedding1 = tf.one_hot(x1_input, vocab_size)

        self.encoder_output1 = lstm_encoder(embedding1, lstm_units=vocab_size, z_dim=self.z_dim)
        decoder_output1 = lstm_decoder_teacher_forced(embedding1, encoder_output1, max_length, lstm_units=vocab_size)

        # Build Autoencoder2
        x2_input = tf.placeholder(tf.int32, [None, None])
        embedding2 = tf.one_hot(x2_input, vocab_size)

        self.encoder_output2 = lstm_encoder(embedding2, lstm_units=vocab_size, z_dim=self.z_dim)
        decoder_output2 = lstm_decoder_teacher_forced(embedding2, encoder_output2, max_length, lstm_units=vocab_size, \
                                                        training=False, softmax=True, temperature=self.temperature)

        # Reconstruction Loss
        self.rec_loss1 = tf.reduce_mean(tf.square(decoder_output1 - embedding1))
        self.rec_loss2 = tf.reduce_mean(tf.square(decoder_output2 - embedding2))

        # Define Discrimator
        with tf.variable_scope("adversary") as scope:
            logits1 = discrimator(encoder_output1)
            logits2 = discrimator(encoder_output2, reuse=True)

        # Adversarial Loss
        _, adv_loss = gan_loss(logits1, logits2)
        
        # Define losses
        self.G_loss = (rec_loss1 + rec_loss2) + self.lambda_g * (adv_loss)
        self.D_loss = adv_loss

        #Define optimizers
        D_solver, G_solver = get_solvers()
        self.G_train_step = G_solver.minimize(G_loss)
        self.D_train_step = D_solver.minimize(D_loss)

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
            self.max_length = X1.shape[1]
            print("Vocab size: {0}".format(self.vocab_size))
            print("Max Length: {0}".format(self.max_length))

    def train(self, pickle_path, num_epochs=10, batch_size=100, \
               saved_model_path='../models/aligned_lstm_ae/aligned_lstm_ae.ckpt' \
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
                num_batches = int(self.X.shape[0] / batch_size)
                for b in range(num_batches):
                    x1_batch = self.X1[b * batch_size:(b + 1) * batch_size]
                    x2_batch = self.X2[b * batch_size:(b + 1) * batch_size]

                    _, _, summary = sess.run([self.D_train_step, self.G_train_step, self.merged], \
                                feed_dict={x1_input: x1_batch, x2_input: x2_batch})

                    step += batch_size

    ########################
    ### Helper Functions ###
    ########################
    def lstm_encoder(x, lstm_units=2, z_dim=10, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        with tf.variable_scope('Encoder'):
            initializer = tf.contrib.layers.xavier_initializer()
            lstm_fw = tf.nn.rnn_cell.LSTMCell(lstm_units, initializer=initializer)
    
            outputs, state = tf.nn.dynamic_rnn(lstm_fw, x, dtype=tf.float32)
            c, h = state
            z = tf.add(tf.layers.dense(c, z_dim), tf.layers.dense(h, z_dim))
            return z

    def lstm_decoder_teacher_forced(x, z, max_length, lstm_units=2, training=True, reuse=False, softmax=True, temperature=0.001):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        with tf.variable_scope('Decoder', reuse=reuse):
            c = tf.layers.dense(z, lstm_units)
            h = tf.layers.dense(z, lstm_units)
            state = tf.nn.rnn_cell.LSTMStateTuple(c, h)
            
            initializer = tf.contrib.layers.xavier_initializer()
            lstm_fw = tf.nn.rnn_cell.LSTMCell(lstm_units, initializer=initializer)
            
            if training:
                with tf.name_scope('Training'):
                    go_vector = tf.fill([tf.shape(x)[0], 1, lstm_units], 0)
                    go_vector = tf.cast(go_vector, tf.float32)
    
                    x_shifted_right = tf.concat([go_vector, x[:,:-1,:]], axis=1)
    
                    outputs, state = tf.nn.dynamic_rnn(lstm_fw, x_shifted_right, initial_state=state)
                    return outputs
            else:
                with tf.name_scope('Inference'):
                    input_tensor = tf.fill([tf.shape(x)[0], 1, lstm_units], 0.0)
                    outputs = []
                    for i in range(max_length):
                        input_tensor, state = tf.nn.dynamic_rnn(lstm_fw, input_tensor, initial_state=state)
                        outputs.append(input_tensor)
                        
                        input_tensor = tf.divide(input_tensor, temperature)

                        # use labels, not softmax
                        if not softmax:
                            input_tensor = tf.argmax(input_tensor, axis=2)
                            input_tensor = tf.one_hot(input_tensor, lstm_units, axis=2)
                    return tf.concat(outputs, axis=1)

    def leaky_relu(x, alpha=0.01):
        """Compute the leaky ReLU activation function.
        
        Inputs:
        - x: TensorFlow Tensor with arbitrary shape
        - alpha: leak parameter for leaky ReLU
        
        Returns:
        TensorFlow Tensor with the same shape as x
        """
        # TODO: implement leaky ReLU
        out = tf.maximum(tf.cast(0.0, dtype='float64'), tf.cast(x, dtype='float64'))
        out1 = tf.minimum(tf.cast(0.0, dtype='float64'), tf.cast(alpha * x, dtype='float64'))
        return tf.cast(out + out1, dtype='float32')
    
    def discriminator(x, reuse=False):
        """Compute discriminator score for a batch of input images.
        
        Inputs:
        - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]
        
        Returns:
        TensorFlow Tensor with shape [batch_size, 1], containing the score 
        for an image being real for each input image.
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()
        with tf.variable_scope("discriminator"):
            # TODO: implement architecture
            x = tf.layers.dense(x, 10)
            x = leaky_relu(x)
            x = tf.layers.dense(x, 10)
            x = leaky_relu(x)
            x = tf.layers.dense(x, 1)
            logits = x
            return logits

    def gan_loss(logits_real, logits_fake):
        """Compute the GAN loss.
        
        Inputs:
        - logits_real: Tensor, shape [batch_size, 1], output of discriminator
            Log probability that the image is real for each real image
        - logits_fake: Tensor, shape[batch_size, 1], output of discriminator
            Log probability that the image is real for each fake image
        
        Returns:
        - D_loss: discriminator loss scalar
        - G_loss: generator loss scalar
        """
        # TODO: compute D_loss and G_loss
        labels_real = tf.ones_like(logits_real)
        labels_fake = tf.zeros_like(logits_fake)
        logits_fake_inv = -logits_fake
        
        D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_real, logits=logits_real) +
                                tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_fake, logits=logits_fake))
        
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_real, logits=logits_fake))
    
        return D_loss, G_loss
    
    def get_solvers(learning_rate=1e-3, beta1=0.5):
        """Create solvers for GAN training.
        
        Inputs:
        - learning_rate: learning rate to use for both solvers
        - beta1: beta1 parameter for both solvers (first moment decay)
        
        Returns:
        - D_solver: instance of tf.train.AdamOptimizer with correct learning_rate and beta1
        - G_solver: instance of tf.train.AdamOptimizer with correct learning_rate and beta1
        """
        D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        return D_solver, G_solver
