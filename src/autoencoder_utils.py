""" Helper functions for autoencoder.py """

import tensorflow as tf
import numpy as np

def lstm_encoder(x, lstm_units=2, z_dim=10, reuse=False, name=''):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.variable_scope('Encoder' + name):
        initializer = tf.contrib.layers.xavier_initializer()
        lstm_fw = tf.nn.rnn_cell.LSTMCell(lstm_units, initializer=initializer)

        outputs, state = tf.nn.dynamic_rnn(lstm_fw, x, dtype=tf.float32)
        c, h = state
        z = tf.add(tf.layers.dense(c, z_dim), tf.layers.dense(h, z_dim))
        return z

def lstm_decoder_teacher_forced(x, z, max_length, lstm_units=2, training=True, reuse=False, softmax=True, temperature=0.001, name=''):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.variable_scope('Decoder' + name, reuse=reuse):
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


