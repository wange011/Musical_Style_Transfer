from abc import ABCMeta, abstractmethod

import tensorflow as tf

def EncodingBlock(X):

    """
    Input of shape (batch_size, 1, 78, timesteps * 2)

    """

    # Kernel with height for 2 octaves, resultant shape: (batch_size, filters = 96, new_rows = 63, new_cols = timesteps)
    conv1 = tf.keras.layers.Conv2D(filters=96, kernel_size=(15, 2), strides=(1, 2), padding="valid", activation=tf.nn.relu)(X)
    
    # Pool each beat together, resultant shape: (batch_size, channels = 96, pooled_rows = 16, pooled_cols = timesteps / 4)
    pool1 = tf.keras.layers.MaxPool2D(pool_size=(4, 4), padding="same", data_format="channels_first")(conv1)
    
    # Kernel with height for 1 octave, resultant shape: (batch_size, filters = 256, new_rows = 9, new_cols = timesteps / 8)
    conv2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(7, 2), strides=(1, 2), padding="valid", activation=tf.nn.relu)(pool1)
    
    # Pool each measure together, resultant shape: (batch_size, channels = 256, pooled_rows = 4, pooled_cols = timesteps / 16)
    pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding="valid", data_format="channels_first")(conv2)
    
    unfold = Unfold(scope='unfold')(pool2)
    output = tf.layers.Dense(2048, activation=tf.nn.relu)(unfold)

    return output

def DecodingBlock(z):

    decoded = tf.layers.Dense()
    fold = Fold([-1, 7, 7, 32], scope='fold')(decoded)
    
    unpool1 = tf.keras.layers.UpSampling2D()(fold)
    
    deconv1 = tf.keras.layers.Conv2DTranspose()(unpool1)
    
    unpool2 = tf.keras.layers.UpSampling2D()(deconv1)
    
    output = tf.keras.layers.Conv2DTranspose()(unpool2)

    return output

class Layer(object, metaclass=ABCMeta):
    """
    """
    def __init__(self):
        pass

    @abstractmethod
    def call(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

class DeConvolution2D(Layer):
    """
    """
    def __init__(self,
                kernel_shape,
                output_shape,
                kernel=None,
                bias=None,
                strides=(1, 1, 1, 1),
                padding='SAME',
                activation=None,
                scope=''):
        Layer.__init__(self)

        self.kernel_shape = kernel_shape
        self.output_shape = output_shape
        self.kernel = kernel
        self.bias = bias
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.scope = scope

    def build(self, input_tensor):
        # build kernel
        if self.kernel:
            assert self.kernel.get_shape() == self.kernel_shape
        else:
            self.kernel = tf.Variable(tf.truncated_normal(self.kernel_shape, stddev=0.1), name='kernel')

        # build bias
        window_height, window_width, num_output_channels, num_input_channels = self.kernel.get_shape()
        if self.bias:
            assert self.bias.get_shape() == (num_output_channels, )
        else:
            self.bias = tf.Variable(tf.constant(0.1, shape=[num_output_channels]), name='bias')

        # convolution
        deconv = tf.nn.conv2d_transpose(input_tensor,
                                        self.kernel,
                                        output_shape=self.output_shape,
                                        strides=self.strides,
                                        padding=self.padding)

        # activation
        if self.activation:
            return self.activation(deconv + self.bias)
        return deconv + self.bias

    def call(self, input_tensor):
        if self.scope:
            with tf.variable_scope(self.scope) as scope:
                return self.build(input_tensor)
        else:
            return self.build(input_tensor)


class UnPooling(Layer):
    """
    Unpool a max-pooled layer.
    Currently this method does not use the argmax information from the previous pooling layer.
    Currently this method assumes that the size of the max-pooling filter is same as the strides.
    Each entry in the pooled map would be replaced with an NxN kernel with the original entry in the upper left.
    For example: a 1x2x2x1 map of
        [[[[1], [2]],
        [[3], [4]]]]
    could be unpooled to a 1x4x4x1 map of
        [[[[ 1.], [ 0.], [ 2.], [ 0.]],
        [[ 0.], [ 0.], [ 0.], [ 0.]],
        [[ 3.], [ 0.], [ 4.], [ 0.]],
        [[ 0.], [ 0.], [ 0.], [ 0.]]]]
    """
    def __init__(self,
                kernel_shape,
                output_shape,
                scope=''):
        Layer.__init__(self)

        self.kernel_shape = kernel_shape
        self.output_shape = output_shape
        self.scope = scope

    def build(self, input_tensor):
        num_channels = input_tensor.get_shape()[-1]
        input_dtype_as_numpy = input_tensor.dtype.as_numpy_dtype()
        kernel_rows, kernel_cols = self.kernel_shape

        # build kernel
        kernel_value = np.zeros((kernel_rows, kernel_cols, num_channels, num_channels), dtype=input_dtype_as_numpy)
        kernel_value[0, 0, :, :] = np.eye(num_channels, num_channels)
        kernel = tf.constant(kernel_value)

        # do the un-pooling using conv2d_transpose
        unpool = tf.nn.conv2d_transpose(input_tensor,
                                        kernel,
                                        output_shape=self.output_shape,
                                        strides=(1, kernel_rows, kernel_cols, 1),
                                        padding='VALID')
        # TODO test!!!
        return unpool

    def call(self, input_tensor):
        if self.scope:
            with tf.variable_scope(self.scope) as scope:
                return self.build(input_tensor)
        else:
            return self.build(input_tensor)


class Unfold(Layer):
    """
    """
    def __init__(self,
                scope=''):
        Layer.__init__(self)

        self.scope = scope

    def build(self, input_tensor):
        num_batch, height, width, num_channels = input_tensor.get_shape()

        return tf.reshape(input_tensor, [-1, (height * width * num_channels).value])

    def call(self, input_tensor):
        if self.scope:
            with tf.variable_scope(self.scope) as scope:
                return self.build(input_tensor)
        else:
            return self.build(input_tensor)


class Fold(Layer):
    """
    """
    def __init__(self,
                fold_shape,
                scope=''):
        Layer.__init__(self)

        self.fold_shape = fold_shape
        self.scope = scope

    def build(self, input_tensor):
        return tf.reshape(input_tensor, self.fold_shape)

    def call(self, input_tensor):
        if self.scope:
            with tf.variable_scope(self.scope) as scope:
                return self.build(input_tensor)
        else:
            return self.build(input_tensor)
