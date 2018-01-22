
# coding: utf-8

# # Face Generation
# In this project, you'll use generative adversarial networks to generate new images of faces.
# ### Get the Data
# You'll be using two datasets in this project:
# - MNIST
# - CelebA
# 
# Since the celebA dataset is complex and you're doing GANs in a project for the first time, we want you to test your neural network on MNIST before CelebA.  Running the GANs on MNIST will allow you to see how well your model trains sooner.
# 
# If you're using [FloydHub](https://www.floydhub.com/), set `data_dir` to "/input" and use the [FloydHub data ID](http://docs.floydhub.com/home/using_datasets/) "R5KrjnANiKVhLWAkpXhNBe".

# In[1]:


data_dir = './data'

# FloydHub - Use with data ID "R5KrjnANiKVhLWAkpXhNBe"
#data_dir = '/input'


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper

helper.download_extract('mnist', data_dir)
helper.download_extract('celeba', data_dir)


# ## Explore the Data
# ### MNIST
# As you're aware, the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset contains images of handwritten digits. You can view the first number of examples by changing `show_n_images`. 

# In[2]:


show_n_images = 25

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
get_ipython().magic('matplotlib inline')
import os
from glob import glob
from matplotlib import pyplot

mnist_images = helper.get_batch(glob(os.path.join(data_dir, 'mnist/*.jpg'))[:show_n_images], 28, 28, 'L')
#pyplot.imshow(helper.images_square_grid(mnist_images, 'L'), cmap='gray')


# In[3]:


pyplot.imshow(helper.images_square_grid(mnist_images, 'L'), cmap='gray')


# ### CelebA
# The [CelebFaces Attributes Dataset (CelebA)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset contains over 200,000 celebrity images with annotations.  Since you're going to be generating faces, you won't need the annotations.  You can view the first number of examples by changing `show_n_images`.

# In[4]:


show_n_images = 25

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
mnist_images = helper.get_batch(glob(os.path.join(data_dir, 'img_align_celeba/*.jpg'))[:show_n_images], 28, 28, 'RGB')
pyplot.imshow(helper.images_square_grid(mnist_images, 'RGB'))


# ## Preprocess the Data
# Since the project's main focus is on building the GANs, we'll preprocess the data for you.  The values of the MNIST and CelebA dataset will be in the range of -0.5 to 0.5 of 28x28 dimensional images.  The CelebA images will be cropped to remove parts of the image that don't include a face, then resized down to 28x28.
# 
# The MNIST images are black and white images with a single [color channel](https://en.wikipedia.org/wiki/Channel_(digital_image%29) while the CelebA images have [3 color channels (RGB color channel)](https://en.wikipedia.org/wiki/Channel_(digital_image%29#RGB_Images).
# ## Build the Neural Network
# You'll build the components necessary to build a GANs by implementing the following functions below:
# - `model_inputs`
# - `discriminator`
# - `generator`
# - `model_loss`
# - `model_opt`
# - `train`
# 
# ### Check the Version of TensorFlow and Access to GPU
# This will check to make sure you have the correct version of TensorFlow and access to a GPU

# In[5]:


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


# ### Input
# Implement the `model_inputs` function to create TF Placeholders for the Neural Network. It should create the following placeholders:
# - Real input images placeholder with rank 4 using `image_width`, `image_height`, and `image_channels`.
# - Z input placeholder with rank 2 using `z_dim`.
# - Learning rate placeholder with rank 0.
# 
# Return the placeholders in the following the tuple (tensor of real input images, tensor of z data)

# In[6]:


import problem_unittests as tests

def model_inputs(image_width, image_height, image_channels, z_dim):
    """
    Create the model inputs
    :param image_width: The input image width
    :param image_height: The input image height
    :param image_channels: The number of image channels
    :param z_dim: The dimension of Z
    :return: Tuple of (tensor of real input images, tensor of z data, learning rate)
    """
    # Tensor of real input images
    input_real = tf.placeholder(
        dtype=tf.float32, 
        shape=(None, image_width, image_height, image_channels), 
        name='input_real')
    
    # Tensor of z data
    input_z = tf.placeholder(
        dtype=tf.float32,
        shape=(None, z_dim), 
        name='input_z')
    
    # Learning rate
    learning_rate = tf.placeholder(dtype=tf.float32)
    
    return input_real, input_z, learning_rate


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_inputs(model_inputs)


# ### Discriminator
# Implement `discriminator` to create a discriminator neural network that discriminates on `images`.  This function should be able to reuse the variables in the neural network.  Use [`tf.variable_scope`](https://www.tensorflow.org/api_docs/python/tf/variable_scope) with a scope name of "discriminator" to allow the variables to be reused.  The function should return a tuple of (tensor output of the discriminator, tensor logits of the discriminator).

# In[7]:


# 不可以指定卷积层与Batch Norm层的name，否则会报错：
# ValueError: Variable conv1/kernel already exists, disallowed. 
# Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?
# 遗留问题。
    
def discriminator(images, reuse=False):
    """
    Create the discriminator network
    :param images: Tensor of input image(s)
    :param reuse: Boolean if the weights should be reused
    :return: Tuple of (tensor output of the discriminator, tensor logits of the discriminator)
    """
    with tf.variable_scope('discriminator', reuse=reuse):
        # Layer 1
        conv1 = tf.layers.conv2d(
            inputs=images, 
            filters=64, 
            kernel_size=5, 
            strides=1,
            padding='same')
        
        lr1 = tf.nn.leaky_relu(features=conv1, name='lk1')
        
        # Layer 2
        conv2 = tf.layers.conv2d(
            inputs=lr1, 
            filters=128, 
            kernel_size=5, 
            strides=2, 
            padding='same')
        
        bn2 = tf.layers.batch_normalization(inputs=conv2, training=True)
        
        lr2 = tf.nn.leaky_relu(features=bn2, name='lk2')
        
        # Layer 3
        conv3 = tf.layers.conv2d(
            inputs=lr2, 
            filters=256, 
            kernel_size=5, 
            strides=2, 
            padding='same')
        
        bn3 = tf.layers.batch_normalization(inputs=conv3, training=True)
        
        lr3 = tf.nn.leaky_relu(features=bn3, name='lk3')
        
        # Layer 4
        flatten = tf.layers.flatten(inputs=lr3, name='flatten')
        logits = tf.layers.dense(inputs=flatten, units=1)
        output = tf.nn.sigmoid(logits)

    return output, logits



"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_discriminator(discriminator, tf)


# ### Generator
# Implement `generator` to generate an image using `z`. This function should be able to reuse the variables in the neural network.  Use [`tf.variable_scope`](https://www.tensorflow.org/api_docs/python/tf/variable_scope) with a scope name of "generator" to allow the variables to be reused. The function should return the generated 28 x 28 x `out_channel_dim` images.

# In[8]:


def generator(z, out_channel_dim, is_train=True, reuse=False):
    """
    Create the generator network
    :param z: Input z
    :param out_channel_dim: The number of channels in the output image
    :param is_train: Boolean if generator is being used for training
    :return: The tensor output of the generator
    """
    if is_train is False:
        reuse = True
        
    with tf.variable_scope('generator', reuse=reuse):
        # Layer 1
        dense1 = tf.layers.dense(inputs=z, units=4*4*512, activation=None)
        seeds = tf.reshape(tensor=dense1, shape=(-1, 4, 4, 512), name='seeds')
        bn1 = tf.layers.batch_normalization(inputs=seeds, training=is_train)
        lk1 = tf.nn.leaky_relu(bn1, name='lk1')
        # [4, 4, 512]
        
        # Layer 2
        conv2 = tf.layers.conv2d_transpose(inputs=lk1, filters=256, kernel_size=4, strides=1, padding='valid')
        bn2 = tf.layers.batch_normalization(inputs=conv2, training=is_train)
        lk2 = tf.nn.leaky_relu(bn2, name='lk2')
        # [7, 7, 256]
        
        # Layer 3
        conv3 = tf.layers.conv2d_transpose(inputs=lk2, filters=128, kernel_size=3, strides=2, padding='same')
        bn3 = tf.layers.batch_normalization(inputs=conv3, training=is_train)
        lk3 = tf.nn.leaky_relu(bn3, name='lk3')
        # [14, 14, 128]

        # Layer 4 -> [28, 28, 3]
        logits = tf.layers.conv2d_transpose(inputs=lk3, filters=out_channel_dim, kernel_size=5, strides=2, padding='same')
        out = tf.tanh(logits)
        
    
    return out


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_generator(generator, tf)


# ### Loss
# Implement `model_loss` to build the GANs for training and calculate the loss.  The function should return a tuple of (discriminator loss, generator loss).  Use the following functions you implemented:
# - `discriminator(images, reuse=False)`
# - `generator(z, out_channel_dim, is_train=True)`

# In[9]:


def model_loss(input_real, input_z, out_channel_dim):
    """
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_z: Z input
    :param out_channel_dim: The number of channels in the output image
    :return: A tuple of (discriminator loss, generator loss)
    """
    # Generate fake image with generator
    image_generated = generator(z= input_z, is_train=True, out_channel_dim=out_channel_dim)
    
    # Discriminator shows up.  Learn from real image..
    estimation_from_real, logtis_real = discriminator(images=input_real, reuse=False)
    # ..then the fake one.
    estimation_from_fake, logits_fake = discriminator(images=image_generated, reuse=True)
    
    loss_d_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logtis_real, labels=tf.ones_like(estimation_from_real)))
    
    loss_d_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits_fake, labels=tf.zeros_like(estimation_from_fake)))
    
    loss_d = loss_d_real + loss_d_fake
    
    loss_g = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits_fake, labels=tf.ones_like(estimation_from_fake)))
    
    return loss_d, loss_g


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_loss(model_loss)


# ### Optimization
# Implement `model_opt` to create the optimization operations for the GANs. Use [`tf.trainable_variables`](https://www.tensorflow.org/api_docs/python/tf/trainable_variables) to get all the trainable variables.  Filter the variables with names that are in the discriminator and generator scope names.  The function should return a tuple of (discriminator training operation, generator training operation).

# In[10]:


def model_opt(d_loss, g_loss, learning_rate, beta1):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tuple of (discriminator training operation, generator training operation)
    """
    var_d = tf.trainable_variables(scope='discriminator')
    var_g = tf.trainable_variables(scope='generator')
    
    # Optimize
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op_d = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(d_loss, var_list=var_d)
        train_op_g = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(g_loss, var_list=var_g)

    return train_op_d, train_op_g


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_opt(model_opt, tf)


# ## Neural Network Training
# ### Show Output
# Use this function to show the current output of the generator during training. It will help you determine how well the GANs is training.

# In[11]:


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

def show_generator_output(sess, n_images, input_z, out_channel_dim, image_mode):
    """
    Show example output for the generator
    :param sess: TensorFlow session
    :param n_images: Number of Images to display
    :param input_z: Input Z Tensor
    :param out_channel_dim: The number of channels in the output image
    :param image_mode: The mode to use for images ("RGB" or "L")
    """
    cmap = None if image_mode == 'RGB' else 'gray'
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

    samples = sess.run(
        generator(input_z, out_channel_dim, False),
        feed_dict={input_z: example_z})

    images_grid = helper.images_square_grid(samples, image_mode)
    pyplot.imshow(images_grid, cmap=cmap)
    pyplot.show()


# ### Train
# Implement `train` to build and train the GANs.  Use the following functions you implemented:
# - `model_inputs(image_width, image_height, image_channels, z_dim)`
# - `model_loss(input_real, input_z, out_channel_dim)`
# - `model_opt(d_loss, g_loss, learning_rate, beta1)`
# 
# Use the `show_generator_output` to show `generator` output while you train. Running `show_generator_output` for every batch will drastically increase training time and increase the size of the notebook.  It's recommended to print the `generator` output every 100 batches.

# In[23]:


def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape, data_image_mode):
    """
    Train the GAN
    :param epoch_count: Number of epochs
    :param batch_size: Batch Size
    :param z_dim: Z dimension
    :param learning_rate: Learning Rate
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :param get_batches: Function to get batches
    :param data_shape: Shape of the data
    :param data_image_mode: The image mode to use for images ("RGB" or "L")
    """
    input_real, input_z, lr = model_inputs(
        image_width=data_shape[1], 
        image_height=data_shape[2], 
        image_channels=data_shape[3], 
        z_dim=z_dim)
    
    loss_d, loss_g = model_loss(input_real=input_real, input_z=input_z, out_channel_dim=data_shape[3])

    train_op_d, train_op_g = model_opt(d_loss=loss_d, g_loss=loss_g, beta1=beta1, learning_rate=learning_rate)
    
    steps = 0
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epoch_count):
            for batch_images in get_batches(batch_size):
                batch_images *= 2
                steps += 1
                
                # Sample random noise for G
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
                
                # Run optimizers
                _ = sess.run(train_op_d, feed_dict={input_real: batch_images, input_z: batch_z})
                _ = sess.run(train_op_g, feed_dict={input_z: batch_z, input_real: batch_images})
                
                if steps % 10 == 0:
                    # At the end of each epoch, get the losses and print them out
                    train_loss_d = loss_d.eval({input_z: batch_z, input_real: batch_images})
                    train_loss_g = loss_g.eval({input_z: batch_z})

                    print("Epoch {}/{}...".format(epoch_i+1, epoch_count),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))
                
                # Show generator output
                if steps % 100 == 0:
                    show_generator_output(
                        image_mode=data_image_mode, 
                        sess=sess, 
                        input_z=input_z, 
                        n_images=batch_size, 
                        out_channel_dim=data_shape[3])
                    
            print('Epoch {} Done!'.format(epoch_i + 1))
        
        show_generator_output(
            image_mode=data_image_mode, 
            sess=sess, 
            input_z=input_z, 
            n_images=batch_size, 
            out_channel_dim=data_shape[3])
        
        print("All done!")    


# ### MNIST
# Test your GANs architecture on MNIST.  After 2 epochs, the GANs should be able to generate images that look like handwritten digits.  Make sure the loss of the generator is lower than the loss of the discriminator or close to 0.

# In[24]:


batch_size = 128
z_dim = 128
learning_rate = 0.01
beta1 = 0.5


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
epochs = 2

mnist_dataset = helper.Dataset('mnist', glob(os.path.join(data_dir, 'mnist/*.jpg')))
with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, mnist_dataset.get_batches,
          mnist_dataset.shape, mnist_dataset.image_mode)


# ### CelebA
# Run your GANs on CelebA.  It will take around 20 minutes on the average GPU to run one epoch.  You can run the whole epoch or stop when it starts to generate realistic faces.

# In[29]:


batch_size = 64
z_dim = 256
learning_rate = 0.001
beta1 = 0.5


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
epochs = 1

celeba_dataset = helper.Dataset('celeba', glob(os.path.join(data_dir, 'img_align_celeba/*.jpg')))
with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, celeba_dataset.get_batches,
          celeba_dataset.shape, celeba_dataset.image_mode)


# ### Submitting This Project
# When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_face_generation.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "helper.py" and "problem_unittests.py" files in your submission.
