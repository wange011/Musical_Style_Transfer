# Sample random piece from style set
# Calculated latent vectors for both style and content
# Perform gradient ascent on content

import tensorflow as tf

import model

training_set, testing_set = utility.loadPianoPieces()

tf.reset_default_graph()

num_notes = 78
X = tf.variable(float, [None, None, num_notes])


z = model.EncodingBlock(X)
output = model.DecodingBlock(z)


# Potentially use a regularizer 
loss = tf.nn.l2_loss(X - output)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)

# Hyperparameters
steps_trained = 
timesteps = 
training_steps = 

training_set = utility.generateBatches(training_set, 1, timesteps)


with tf.Session() as sess:
        
        saver.restore(sess, working_directory + "/saved_models/" + model_name + "_" + str(steps_trained) + "_iterations.ckpt")

        for step in range(1, training_steps + 1):
            
            # Shuffle the training set between each epoch
            if step % num_batches == 1 and step != 1:
                training_set = utility.shuffleBatches(training_set)
            
            batch = training_set[(step - 1) % num_batches]
            
            # Code for X

            # Evaluate the computational graph
            z_run = sess.run([z], feed_dict={X: x})                
            
            # Each display_step iterations, save the model and generate outputs
            if step % display_step == 0:
                
                # Saves the model            
                saver.save(sess, working_directory + "/saved_models/" + model_name + "_" + str(step) + "_iterations.ckpt")            
                
                # Calculate batch loss and accuracy
                print("Step " + str(step) + ", Loss= " + str(loss_run))
                file.write("Step " + str(step) + ", Loss= " + str(loss_run) + "\n")