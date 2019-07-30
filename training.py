import tensorflow as tf

import numpy as np

import os
import utility

def train(model_name, training_set, X, output, loss, train_op, training_parameters):

    working_directory = os.getcwd()
    
    timesteps = training_parameters["timesteps"]
    batch_size = training_parameters["batch_size"]
    training_steps = training_parameters["training_steps"]
    display_step = training_parameters["display_step"]    
        
    # Initialize the variables for the computational graph
    init = tf.global_variables_initializer()
    
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    
    file = open("training_progress.txt","w")    
    
    with tf.Session() as sess:
        
        file.write("Starting Training")
        sess.run(init)
        
        # Generate mini-batches
        # Get batch of shape [batch_size, timesteps, 78, 2]
        training_set = utility.generateBatches(training_set, batch_size, timesteps)

        num_batches = training_set.shape[0]

        for step in range(1, training_steps + 1):
            
            # Shuffle the training set between each epoch
            if step % num_batches == 1 and step != 1:
                training_set = utility.shuffleBatches(training_set)
            
            batch = training_set[(step - 1) % num_batches]
            
            x = np.reshape(batch, (batch_size, 1, timesteps, 78, 2))
            x = np.transpose(x, (0, 1, 3, 2, 4))
            x = np.reshape(x, (batch_size, 1, 78, timesteps * 2))

            # Evaluate the computational graph
            loss_run, outputs_run, _, = sess.run([loss, output, train_op], feed_dict={X: x})                
            
            # Each display_step iterations, save the model and generate outputs
            if step % display_step == 0:
                
                # Saves the model            
                saver.save(sess, working_directory + "/saved_models/" + model_name + "_" + str(step) + "_iterations.ckpt")            
            	


                # Output the reconstructed piece
                # Maybe use a validation set

                # Reshapes the output to the piano roll format
                outputs_run = np.reshape(outputs_run, (batch_size, 1, 78, timesteps, 2))
                outputs_run = np.transpose(outputs_run, (0, 1, 3, 2, 4))
                outputs_run = np.reshape(outputs_run, (batch_size, timesteps, 78, 2))

                for j in range(len(outputs_run)):
                    utility.generateMIDI(outputs_run[j], model_name + "_" + str(step) + "_iterations_" + str(j + 1))

                # Calculate batch loss and accuracy
                print("Step " + str(step) + ", Loss= " + str(loss_run))
                file.write("Step " + str(step) + ", Loss= " + str(loss_run) + "\n")
