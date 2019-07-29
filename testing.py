import tensorflow as tf

import numpy as np

def train(model_name, testing_set, X, output, loss, train_op, training_parameters):
	working_directory = os.getcwd()
    
    timesteps = training_parameters["timesteps"]
    batch_size = training_parameters["batch_size"]
        
    # Initialize the variables for the computational graph
    init = tf.global_variables_initializer()
    
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    
    file = open("testing_accuracy.txt","w")    
    
    with tf.Session() as sess:
        
        file.write("Testing")
        sess.run(init)
        
        # Generate mini-batches
        # Get batch of shape [batch_size, timesteps, 78, 2]
        testing_set = utility.generateBatches(testing_set, batch_size, timesteps)

        num_batches = testing_set.shape[0]

        for step in range(num_batches):
            
            
            batch = testing_set[(step - 1) % num_batches]
            
            x = np.reshape(batch, (batch_size, 1, timesteps, 78, 2))
            x = np.transpose(x, (0, 1, 3, 2, 4))
            x = np.reshape(x, (batch_size, 1, 78, timesteps * 2))

            # Evaluate the computational graph
            loss_run, outputs_run = sess.run([loss, outputs], feed_dict={X: x})                
            
            # Each display_step iterations, save the model and generate outputs
            if step % 5 == 0:
                
                # Output the reconstructed piece
                # Maybe use a validation set

                # Reshapes the output to the piano roll format
                outputs_run = np.reshape(outputs_run, (batch_size, 1, 78, timesteps, 2))
                outputs_run = np.transpose(outputs_run, (0, 1, 3, 2, 4))
                outputs_run = np.reshape(outputs_run, (batch_size, timesteps, 78, 2))

                for j in range(len(outputs_run)):
                    utility.generateMIDI(outputs_run[j], model_name + "_" + str(step) + "_iterations_" + str(j + 1))

                # Calculate batch loss and accuracy
                print("Loss= " + str(loss_run))
                file.write("Loss= " + str(loss_run) + "\n")
