#This file contains various network Archetictures that were examined


def CarSign_Model_1(x):    
    
    """
        * Two Convolution Layers followed by 2 flattening layers and a final output layer
        * Achieves 96% training accuracy after 50 epoch and 75% validation 
        
    """
    mu = 0
    sigma = 0.1

    weights = {

    # Layer 1: Convolutional. Input = 32x32x1. Output = 30x30x24.
    'conv1_W': tf.Variable(tf.truncated_normal(shape=(3, 3, 1, 24), mean = mu, stddev = sigma)),
    # Layer 2: Convolutional. Input 15x15x24, Output = 10x10x32.
    'conv2_W': tf.Variable(tf.truncated_normal(shape=(5, 5, 24, 32), mean = mu, stddev = sigma)),
    # Layer 3: Fully Connected. Input = 5x5x32 = 800. Output = 600.
    'fc1_W' : tf.Variable(tf.truncated_normal(shape=(800, 600), mean = mu, stddev = sigma)),
    # Layer 4: Fully Connected. Input = 600. Output = 256.
    'fc2_W'  : tf.Variable(tf.truncated_normal(shape=(600, 256), mean = mu, stddev = sigma)),
    # Layer 5: Output, Fully Connected. Input = 256. Output = 43.  
    'out' : tf.Variable(tf.truncated_normal(shape=(256, n_classes), mean = mu, stddev = sigma))

    }

    biases = {
        'conv1_b': tf.Variable(tf.random_normal([24])),
        'conv2_b': tf.Variable(tf.random_normal([32])),
        'fc1_b': tf.Variable(tf.random_normal([600])),
        'fc2_b': tf.Variable(tf.random_normal([256])),
        'out_b': tf.Variable(tf.random_normal([n_classes]))
    }


    conv1 = conv_layer(x, weights['conv1_W'], biases['conv1_b'],my_padding='VALID')
    #conv1 = tf.nn.dropout(conv1, dropout)
    conv1 = max_pool_layer(conv1, k=2)
    
    conv2   = conv_layer(conv1,weights['conv2_W'], biases['conv2_b'] )
    #conv2 = tf.nn.dropout(conv2, dropout)
    conv2 = max_pool_layer(conv2, k=2)
    
    
    fc0   = flatten(conv2) # Flatten. Input = 5x5x32. Output = 800.
    
    fc1 = tf.add(tf.matmul(fc0, weights['fc1_W']), biases['fc1_b'])
    fc1 = tf.nn.relu(fc1)
    
    fc2 = tf.add(tf.matmul(fc1, weights['fc2_W']), biases['fc2_b'])
    fc2 = tf.nn.relu(fc2)
    
    #fc2 = tf.nn.dropout(fc2, dropout)
    

    # Output, class prediction
    logits = tf.add(tf.matmul(fc2, weights['out']), biases['out_b'])
    
 
    
    return logits
"""***********************************************************************"""
"""******************************************************************************************************************"""

# Store layers weight & bias

mu = 0
sigma = 0.1

weights = {

    # Layer 1: Convolutional. Input = 32x32x1. Output = 30x30x16.
    'conv1_W': tf.Variable(tf.truncated_normal(shape=(3, 3, 1, 16), mean = mu, stddev = sigma)),
    # Layer 2: Convolutional. Input 15x15x16, Output = 12x12x24.
    'conv2_W': tf.Variable(tf.truncated_normal(shape=(4, 4, 16, 24), mean = mu, stddev = sigma)),
    # Layer 3: Convolutional. Input 6x6x24, Output = 6x6x32.
    'conv3_W': tf.Variable(tf.truncated_normal(shape=(1, 1, 24, 32), mean = mu, stddev = sigma)),
    # Layer 4: Fully Connected. Input = 864. Output = 600.
    'fc1_W' : tf.Variable(tf.truncated_normal(shape=(864, 600), mean = mu, stddev = sigma)),
    # Layer 5: Fully Connected. Input = 600. Output = 256.
    'fc2_W'  : tf.Variable(tf.truncated_normal(shape=(600, 256), mean = mu, stddev = sigma)),
    # Layer 6: Fully Connected. Input = 5x5x32 = 800. Output = 600.
    'fc3_W' : tf.Variable(tf.truncated_normal(shape=(256, 128), mean = mu, stddev = sigma)),
    # Layer 7: Fully Connected. Input = 600. Output = 256.
    'fc4_W'  : tf.Variable(tf.truncated_normal(shape=(128, 96), mean = mu, stddev = sigma)),
    # Layer 5: Output, Fully Connected. Input = 256. Output = 43.  
    'out' : tf.Variable(tf.truncated_normal(shape=(96, n_classes), mean = mu, stddev = sigma))

}

biases = {
    'conv1_b': tf.Variable(tf.random_normal([16])),
    'conv2_b': tf.Variable(tf.random_normal([24])),
    'conv3_b': tf.Variable(tf.random_normal([32])),
    'fc1_b': tf.Variable(tf.random_normal([600])),
    'fc2_b': tf.Variable(tf.random_normal([256])),
    'fc3_b': tf.Variable(tf.random_normal([128])),
    'fc4_b': tf.Variable(tf.random_normal([96])),
    'out_b': tf.Variable(tf.random_normal([n_classes]))
}

"""
    * Two Convolution Layers followed by 2 flattening layers and a final output layer
    * Achieves 96% training accuracy after 50 epoch and 75% validation 
"""

conv1 = conv_layer(x, weights['conv1_W'], biases['conv1_b'],my_padding='VALID')
conv1 = tf.nn.dropout(conv1, dropout)
conv1 = max_pool_layer(conv1, k=2)

conv2   = conv_layer(conv1,weights['conv2_W'], biases['conv2_b'] )
#conv2 = tf.nn.dropout(conv2, dropout)
conv2 = max_pool_layer(conv2, k=2)

conv3 = conv_layer(conv2, weights['conv3_W'], biases['conv3_b'],my_padding='SAME')
conv3 = max_pool_layer(conv3, k=2)


fc0   = flatten(conv2) # Flatten. Input = 5x5x32. Output = 800.

fc1 = tf.add(tf.matmul(fc0, weights['fc1_W']), biases['fc1_b'])
fc1 = tf.nn.relu(fc1)

fc2 = tf.add(tf.matmul(fc1, weights['fc2_W']), biases['fc2_b'])
fc2 = tf.nn.relu(fc2)

fc3 = tf.add(tf.matmul(fc2, weights['fc3_W']), biases['fc3_b'])
fc3 = tf.nn.relu(fc3)

fc4 = tf.add(tf.matmul(fc3, weights['fc4_W']), biases['fc4_b'])
fc4 = tf.nn.relu(fc4)

# Output, class prediction
logits = tf.add(tf.matmul(fc4, weights['out']), biases['out_b'])



"""******************************************************************************************************************"""
"""------------------------------------------------------------------------------------------------------------------"""

# Store layers weight & bias

mu = 0
sigma = 0.1

weights = {

    
    'norm_conv1'  : tf.Variable(tf.truncated_normal(shape=(3, 3, 3, 4), mean = mu, stddev = sigma)),
    'norm_conv2'  : tf.Variable(tf.truncated_normal(shape=(3, 3, 2, 4), mean = mu, stddev = sigma)),
    
    'incep1_Conv_1x1_W'  : tf.Variable(tf.truncated_normal(shape=(1, 1, 4, 8), mean = mu, stddev = sigma)),
    'incep1_Conv_3x3_W'  : tf.Variable(tf.truncated_normal(shape=(3, 3, 4, 8), mean = mu, stddev = sigma)),
    'incep1_Conv_5x5_W'  : tf.Variable(tf.truncated_normal(shape=(5, 5, 4, 8), mean = mu, stddev = sigma)),
    
    'incep2_Conv_1x1_W'  : tf.Variable(tf.truncated_normal(shape=(1, 1, 8, 8), mean = mu, stddev = sigma)),
    'incep2_Conv_3x3_W'  : tf.Variable(tf.truncated_normal(shape=(3, 3, 8, 8), mean = mu, stddev = sigma)),
    'incep2_Conv_5x5_W'  : tf.Variable(tf.truncated_normal(shape=(5, 5, 8, 8), mean = mu, stddev = sigma)),
    
    
    'norm_conv3'  : tf.Variable(tf.truncated_normal(shape=(3, 3, 8, 16), mean = mu, stddev = sigma)),
    
    'Full_C1'  : tf.Variable(tf.truncated_normal(shape=(400, 400), mean = mu, stddev = sigma)),
    'Full_C2'  : tf.Variable(tf.truncated_normal(shape=(400,86), mean = mu, stddev = sigma)),
    'output'   : tf.Variable(tf.truncated_normal(shape=(86, n_classes), mean = mu, stddev = sigma))

}

biases = {

    'norm_conv1_b': tf.Variable(tf.random_normal([4])),
    'norm_conv2_b': tf.Variable(tf.random_normal([4])),
    
    'incep1_Conv_bias': tf.Variable(tf.random_normal([8])),
    'incep2_Conv_bias': tf.Variable(tf.random_normal([8])),
    #'incep3_Conv_bias': tf.Variable(tf.random_normal([16])),
    
    'norm_conv3_b': tf.Variable(tf.random_normal([16])),


    'fc1_b': tf.Variable(tf.random_normal([400])),
    'fc2_b': tf.Variable(tf.random_normal([86])),

    'output_b': tf.Variable(tf.random_normal([n_classes]))
}


"""
   #From Alex NET Example
   # lrn2
    # lrn(2, 2e-05, 0.75, name='norm2')

"""



print(x.get_shape().as_list() )


normal_conv1 = conv_layer(x,weights['norm_conv1'], biases['norm_conv1_b'],'VALID',1, False )
#normal_conv1 = tf.nn.dropout(normal_conv1, dropout)
#normal_conv1 = max_pool_layer(normal_conv1, k=2)
#---------------------------------------------

#14x14x12 ---> 28x28x64
#normal_conv2 = conv_layer(normal_conv1,weights['norm_conv2'], biases['norm_conv2_b'],'VALID',1, False )
#normal_conv2 = max_pool_layer(normal_conv2, k=2)
#---------------------------------------------


incep1_Conv_1 = conv_layer(normal_conv1, weights['incep1_Conv_1x1_W'], biases['incep1_Conv_bias'], my_padding='SAME',strides=1) 
incep1_Conv_3 = conv_layer(normal_conv1, weights['incep1_Conv_3x3_W'], biases['incep1_Conv_bias'], my_padding='SAME',strides=1) 
incep1_Conv_5 = conv_layer(normal_conv1, weights['incep1_Conv_5x5_W'], biases['incep1_Conv_bias'], my_padding='SAME',strides=1) 

#Input = 12x12x64. Output = 4x4x42,
conv_A = avg_pool_layer(incep1_Conv_1+incep1_Conv_1+incep1_Conv_1, k=2)

incep2_Conv_1 = conv_layer(conv_A, weights['incep2_Conv_1x1_W'], biases['incep2_Conv_bias'], 'SAME',strides=1) 
incep2_Conv_3 = conv_layer(conv_A, weights['incep2_Conv_3x3_W'], biases['incep2_Conv_bias'], 'SAME',strides=1) 
incep2_Conv_5 = conv_layer(conv_A, weights['incep2_Conv_5x5_W'], biases['incep2_Conv_bias'], 'SAME',strides=1) 

conv_B = avg_pool_layer(incep2_Conv_1 + incep2_Conv_3 + incep2_Conv_5, k=2)




#---------------------------------------------

#14x14x64 ---> 24x24x64
normal_conv3 = conv_layer(conv_B,weights['norm_conv3'], biases['norm_conv3_b'],'VALID',1, True )
normal_conv3 = max_pool_layer(normal_conv3, k=1)
normal_conv3 = tf.nn.dropout(normal_conv3, dropout)


#--------------------------------------------------------------------------------
# Layer 1: Convolutional. Using Inception Model
#          Input = 12x12x32. Output = 12x12x42,  Strid. 1
#--------------------------------------------------------------------------------


#7*7*64




#--------------------------------------------------------------------------------
# Layer 4: Fully connected
#          Input = 7*7*16 = 784
#--------------------------------------------------------------------------------

flatten_layer   = flatten(normal_conv3)


fc1 = tf.add(tf.matmul(flatten_layer, weights['Full_C1'] ), biases['fc1_b'])
fc1 = tf.nn.relu(fc1)
#fc1 = tf.nn.softmax(fc1)


#--------------------------------------------------------------------------------
# Layer 5: Fully Connected. Input = 2400. Output = 215
#--------------------------------------------------------------------------------

fc2 = tf.add(tf.matmul(fc1, weights['Full_C2']), biases['fc2_b'])
fc2 = tf.nn.relu(fc2)
fc2 = tf.nn.dropout(fc2, dropout)



logits = tf.add(tf.matmul(fc2, weights['output']), biases['output_b'])




#--------------------------------------------------------------------------------
# Layer 4: Fully connected
#          Input = 7*7*16 = 784
#--------------------------------------------------------------------------------

flatten_layer   = flatten(normal_conv3)


fc1 = tf.add(tf.matmul(flatten_layer, weights['Full_C1'] ), biases['fc1_b'])
fc1 = tf.nn.relu(fc1)
#fc1 = tf.nn.softmax(fc1)


#--------------------------------------------------------------------------------
# Layer 5: Fully Connected. Input = 2400. Output = 215
#--------------------------------------------------------------------------------

fc2 = tf.add(tf.matmul(fc1, weights['Full_C2']), biases['fc2_b'])
fc2 = tf.nn.relu(fc2)
fc2 = tf.nn.dropout(fc2, dropout)



logits = tf.add(tf.matmul(fc2, weights['output']), biases['output_b'])

    




"""******************************************************************************************************************"""
"""------------------------------------------------------------------------------------------------------------------"""

   
    """
    resh = tf.reshape(fc2, [-1,14,14,6]) #-1 means consider all inputs that are none
    print("Reshaped.. ",resh.get_shape().as_list() )
    """
    
   
    """
    conv4   = conv_layer(conv3,weights['conv_4'], biases['conv4_b'] )
    conv4 = tf.nn.dropout(conv4, dropout)
    conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
   
    """
    


"""******************************************************************************************************************"""
"""------------------------------------------------------------------------------------------------------------------"""



