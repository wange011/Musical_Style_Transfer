import tensorflow as tf

import os 
import utility
import model
import training

working_directory = os.getcwd()

# MIDI files are first converted to a noteStateMatrix

"""
Note State:

Dimension 1: Timesteps, converted to a single 16th note per timestep
Dimension 2: Notes played, Piano Roll notation contained in a 78 dimension 1-hot vector (MIDI values are truncated between 24 and 102)
Dimension 3: Articulation (1 denotes the note was played at the given timestep), contained in a 78 dimension 1-hot vector

"""

# Gather the training pieces from the specified directories
# Converts the pieces from noteStateMatrix to the biaxialInputFormat
training_set, testing_set = utility.loadPianoPieces()
scales = utility.loadScales()   

tf.reset_default_graph()

num_notes = 78
X = tf.variable(float, [None, None, num_notes])


z = model.EncodingBlock(X)
output = model.DecodingBlock(z)


# Potentially use a regularizer 
loss = tf.nn.l2_loss(X - output)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)

# Training parameters
model_name = "ConvolutionalAutoencoder"
timesteps = 128
batch_size = 10
steps = 80000
display_step = 1000

# Training the model
training_parameters = {"timesteps": timesteps, "batch_size": batch_size, "training_steps": steps, "display_step": display_step}

training.train(model_name, training_set, X, output, loss, train_op, training_parameters)