import numpy as np
import os
import tensorflow as tf
import urllib2

from datasets import imagenet
from nets import inception_resnet_v2
from preprocessing import inception_preprocessing as preprocessing

def run(image_string):

    checkpoints_dir = '/root/Desktop/LogoReader/data/slim_test/training/inception_resnet_v2'
    checkpoints_file_name = tf.train.latest_checkpoint(checkpoints_dir) 

    slim = tf.contrib.slim

    # We need default size of image for a particular network.
    # The network was trained on images of that size -- so we
    # resize input image later in the code.
    image_size = inception_resnet_v2.inception_resnet_v2.default_image_size


    with tf.Graph().as_default():
      
        # Decode string into matrix with intensity values
        image = tf.image.decode_jpeg(image_string, channels=3)
        
        # Resize the input image, preserving the aspect ratio
        # and make a central crop of the resulted image.
        # The crop will be of the size of the default image size of
        # the network.
        processed_image = preprocessing.preprocess_image(image,
                                                             image_size,
                                                             image_size,
                                                             is_training=False)
        
        # Networks accept images in batches.
        # The first dimension usually represents the batch size.
        # In our case the batch size is one.
        processed_images  = tf.expand_dims(processed_image, 0)
        
        # Create the model, use the default arg scope to configure
        # the batch norm parameters. arg_scope is a very conveniet
        # feature of slim library -- you can define default
        # parameters for layers -- like stride, padding etc.
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits, _ = inception_resnet_v2.inception_resnet_v2(processed_images,
                                   num_classes=51,
                                   is_training=False)
        
        # In order to get probabilities we apply softmax on the output.
        probabilities = tf.nn.softmax(logits)
        
        # Create a function that reads the network weights
        # from the checkpoint file that you downloaded.
        # We will run it in session later.
        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, checkpoints_file_name),
            slim.get_model_variables('InceptionResnetV2'))
        
        with tf.Session() as sess:
            
            # Load weights
            init_fn(sess)
            
            # We want to get predictions, image as numpy matrix
            # and resized and cropped piece that is actually
            # being fed to the network.
            np_image, network_input, probabilities = sess.run([image,
                                                               processed_image,
                                                               probabilities])
            probabilities = probabilities[0, 0:]
            sorted_inds = [i[0] for i in sorted(enumerate(-probabilities),
                                                key=lambda x:x[1])]

        print('index: %d' % sorted_inds[0])

        names = imagenet.create_readable_names_for_imagenet_labels()
        for i in range(5):
            index = sorted_inds[i]
            # Now we print the top-5 predictions that the network gives us with
            # corresponding probabilities. Pay attention that the index with
            # class names is shifted by 1 -- this is because some networks
            # were trained on 1000 classes and others on 1001. VGG-16 was trained
            # on 1000 classes.
            print('Probability %0.2f => [%s]' % (probabilities[index], names[index]))
            
        # res = slim.get_model_variables()

if __name__ == '__main__':
    # img_path = '/root/Downloads/CarLogos51/Benz/0033.jpg'
    # image_string = open(img_path).read()
    img_url = '\
            https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1512648031102&di=bdea805db6d13c9592372549fa003d68&imgtype=0&src=http%3A%2F%2Fpic32.photophoto.cn%2F20140707%2F0022005522358955_b.jpg'
    image_string = urllib2.urlopen(img_url).read()
    run(image_string)
