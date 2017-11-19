# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import os

def preprocess_image_randomly(image_raw_data):
    image_data = tf.image.decode_jpeg(image_raw_data)
    if image_data.dtype != tf.float32:
        image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)
    
    # 随机调整
    image_data = tf.image.random_flip_left_right(image_data)
    image_data = distort_color(image_data)

    image_data = tf.image.convert_image_dtype(image_data, dtype=tf.uint8)
    image_data = tf.image.encode_jpeg(image_data)

    with tf.Session() as sess:
        return image_data.eval()

def distort_color(image):
    image = tf.image.random_brightness(image, max_delta=np.random.uniform(0, 0.5))
    image = tf.image.random_saturation(image, lower=np.random.uniform(0.5, 1.0), upper=np.random.uniform(1.0, 1.5))
    image = tf.image.random_hue(image, max_delta=np.random.uniform(0, 0.5))
    image = tf.image.random_contrast(image, lower=np.random.uniform(0.5, 1.0), upper=np.random.uniform(1.0, 1.5))

    return tf.clip_by_value(image, 0.0, 1.0)

if __name__ == '__main__':
    image_dir = '../../data/logos/丰田'
    images = os.listdir(image_dir)
    for i in range(5):
        image_path = os.path.join(image_dir, images[np.random.randint(len(images))])
        image_raw_data = gfile.FastGFile(image_path, 'rb').read()
        result = preprocess_image_randomly(image_raw_data)
        # image_data = tf.image.decode_jpeg(image_raw_data)
        # if image_data.dtype != tf.float32:
        #     image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)
        # plt.imshow(image_data)
        # plt.show()
        # plt.imshow(result)
        # plt.show()
