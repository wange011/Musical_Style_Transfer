from abc import ABCMeta, abstractmethod

import tensorflow as tf

def EncodingBlock(X, batch_size, timesteps):

    """
    Input of shape (batch_size, 1, 78, timesteps * 2)

    """

    # Kernel with height for 2 octaves, resultant shape: (batch_size, filters = 96, new_rows = 63, new_cols = timesteps)
    conv1 = tf.keras.layers.Conv2D(filters=96, kernel_size=(15, 2), strides=(1, 2), padding="valid", data_format="channels_first", activation=tf.nn.relu)(X)
    
    # Pool each beat together, resultant shape: (batch_size, channels = 96, pooled_rows = 16, pooled_cols = timesteps / 4)
    pool1 = tf.keras.layers.MaxPool2D(pool_size=(4, 4), padding="same", data_format="channels_first")(conv1)
    
    # Kernel with height for 1 octave, resultant shape: (batch_size, filters = 256, new_rows = 9, new_cols = timesteps / 8)
    conv2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(7, 2), strides=(1, 2), padding="valid", data_format="channels_first", activation=tf.nn.relu)(pool1)
    
    # Pool each measure together, resultant shape: (batch_size, channels = 256, pooled_rows = 5, pooled_cols = timesteps / 16)
    pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding="same", data_format="channels_first")(conv2)
    
        
    unfold_dim = tf.cast(256 * 5 * timesteps / 16, tf.int32)
   
    
    # Unfolds the image representation of the piece to a 1D tensor
    unfold = tf.reshape(pool2, [batch_size, unfold_dim])

    # Outputs in shape (batch_size, latent_vector_dim = 2048)
    output = tf.layers.Dense(2048, activation=tf.nn.relu)(unfold)

    return output

def DecodingBlock(z, batch_size, timesteps):

    # Outputs the reconstructed image representation of the piece as a 1D tensor, resultant shape: (batch_size, unfolded_shape = 256 * 5 * timesteps / 16)
    decoded = tf.layers.Dense(256 * 5 * timesteps / 16, activation=tf.nn.relu)(z)

    # Refolds the image representation, resultant shape: (batch_size, channels = 256, pooled_rows = 5, pooled_cols = timesteps / 16)
    fold = tf.reshape(decoded, [batch_size, 256, 5, tf.cast(timesteps / 16, tf.int32)])

    # Unpools, resultant shape: (batch_size, channels = 256, pooled_rows = 10, pooled_cols = timesteps / 8)   
    unpool1 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format="channels_first")(fold)
    
    # Upconvolves, resultant shape: (batch_size, filters = 96, new_rows = 16, new_cols = timesteps / 4)
    upconv1 = tf.keras.layers.Conv2DTranspose(filters=96, kernel_size=(7, 2), strides=(1, 2), padding="valid", data_format="channels_first", activation=tf.nn.relu)(unpool1)
    
    # Unpools, resultant shape: (batch_size, channels = 96, pooled_rows = 64, pooled_cols = timesteps)
    unpool2 = tf.keras.layers.UpSampling2D(size=(4, 4), data_format="channels_first")(upconv1)
    
    # Upconvolves, resultant shape: (batch_size, filters = 1, new_rows = 78, new_cols = timesteps * 2)
    output = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(15, 2), strides=(1, 2), padding="valid", data_format="channels_first", activation=tf.nn.relu)(unpool2)

    return output

# Define VAE Loss
def vae_loss(x_reconstructed, x_true):
    # Reconstruction loss
    encode_decode_loss = x_true * tf.log(1e-10 + x_reconstructed) \
                         + (1 - x_true) * tf.log(1e-10 + 1 - x_reconstructed)
    encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)
    # KL Divergence loss
    kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
    return tf.reduce_mean(encode_decode_loss + kl_div_loss)