# -*- coding:utf-8 -*-

import glob
import os.path
import random
import numpy as np
import tensorflow as tf
import cPickle
from tensorflow.python.platform import gfile
import image_processing
import pdb
import logging

DATASET = 'Binarization_CarLogo51'
CACHE_DIR = '../../data/bottleneck/' + DATASET
# INPUT_DATA = '../../data/' + DATASET
INPUT_DATA = '/root/Desktop/MX/' + DATASET

CLASS_INDEX_FILE = '../../data/class_index.txt'

IMAGE_PROCESSING = False

# 每一张原图对应的处理后的图片数量(包括原图)
NUM_DISTORTED = 5


# 把样本中所有的图片列表并按训练、验证、测试数据分开
def create_image_lists(testing_percentage, validation_percentage):

    result = {}
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']

        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue

        label_name = dir_name.lower()
        
        # 初始化
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            
            # 随机划分数据
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
            }
    # 将分类的索引数组保存下来
    with open(CLASS_INDEX_FILE, 'wb') as f:
        cPickle.dump(result.keys(), f)

    return result

# 定义函数通过类别名称、所属数据集和图片编号获取一张图片的地址。
def get_image_path(image_lists, image_dir, label_name, index, category):
    label_lists = image_lists[label_name]
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path

# 定义函数获取Inception-v3模型处理之后的特征向量的文件地址。
def get_bottleneck_path(image_lists, label_name, index, category, distorted):
    if distorted == 0:
        return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + '.txt'
    else:
        return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + str(distorted) + '.txt'

# 定义函数使用加载的训练好的Inception-v3模型处理一张图片，得到这个图片的特征向量。
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor, distorted):
    if not distorted == 0:
        image_data = image_processing.preprocess_image_randomly(image_data)
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

# 定义函数会先试图寻找已经计算且保存下来的特征向量，如果找不到则先计算这个特征向量，然后保存到文件。
def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path): os.makedirs(sub_dir_path)

    if not IMAGE_PROCESSING:
    	distorted = 0
    else:
   	distorted = np.random.randint(NUM_DISTORTED)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category, distorted)

    if not os.path.exists(bottleneck_path):
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)
        image_data = gfile.FastGFile(image_path, 'rb').read()
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor, distorted)
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    return bottleneck_values

# 随机获取一个batch的图片作为训练数据。
def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many, category, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    counter = 0
    while counter < how_many:
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        # 处理无效图片数据导致的异常
        try:
            bottleneck = get_or_create_bottleneck(
                sess, image_lists, label_name, image_index, category, jpeg_data_tensor, bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
            counter += 1
        # except tf.errors.InvalidArgumentError, e:
        #     pass
        # except tf.errors.FailedPreconditionError, e:
        # 	pass
        except Exception, e:
            pass

    return bottlenecks, ground_truths


# 获取全部的测试数据，并计算正确率。
def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    for label_index, label_name in enumerate(label_name_list):
        # 处理无效图片数据导致的异常
        try:
            category = 'testing'
            for index, unused_base_name in enumerate(image_lists[label_name][category]):
                bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor)
                ground_truth = np.zeros(n_classes, dtype=np.float32)
                ground_truth[label_index] = 1.0
                bottlenecks.append(bottleneck)
                ground_truths.append(ground_truth)
        # except tf.errors.InvalidArgumentError, e:
        #     pass
        # except tf.errors.FailedPreconditionError, e:
        # 	pass
        except Exception, e:
            pass
    return bottlenecks, ground_truths
