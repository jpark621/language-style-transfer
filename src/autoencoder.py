import tensorflow as tf

class Aligned_LSTM_Autoencoder:
    def __init__(self, learning_rate=0.0001, temperature=0.001, beta1=0.9, beta2=0.9, z_dim=100, lambda_g=1):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.z_dim = z_dim
        self.lambda_g = lambda_g
        self.temperature = temperature

    def graph_init(self, vocab_size, max_length):
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

        
