# -*- coding: utf-8 -*-
import tensorflow as tf
import transfer_learning
import requests
import os
import numpy as np
import cPickle
from tensorflow.python.platform import gfile

def main():
    img_url = raw_input('Input an url:\n')
    image_data = requests.get(img_url, timeout=3).content

    with open(transfer_learning.CLASS_INDEX_FILE, 'rb') as f:
        classes = cPickle.load(f)
    num_classes = len(classes)

    with tf.Session() as sess:
        weights = tf.Variable(tf.zeros([transfer_learning.BOTTLENECK_TENSOR_SIZE, num_classes]), name='other_weights')
        biases = tf.Variable(tf.zeros([num_classes]), name='other_biases')
        saver = tf.train.Saver({'final_training_ops/weights': weights, 'final_training_ops/biases': biases})
        
        ckpt = tf.train.get_checkpoint_state(transfer_learning.MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

            # 读取已经训练好的Inception-v3模型。
            with gfile.FastGFile(os.path.join(transfer_learning.MODEL_DIR, transfer_learning.MODEL_FILE), 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
            bottleneck_tensor, image_data_tensor = tf.import_graph_def(
                graph_def, return_elements=[transfer_learning.BOTTLENECK_TENSOR_NAME, transfer_learning.JPEG_DATA_TENSOR_NAME])
            # Compute output
            bottleneck_input = tf.placeholder(tf.float32, [None, transfer_learning.BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
            logits = tf.matmul(bottleneck_input, weights) + biases
            final_tensor = tf.nn.softmax(logits)
            label_index = tf.cast((tf.argmax(final_tensor, 1)), tf.int32)

            bottleneck_values = np.reshape(
                transfer_learning.run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor),
                (1, transfer_learning.BOTTLENECK_TENSOR_SIZE)
                )
            result = classes[ int(sess.run(label_index, feed_dict={bottleneck_input: bottleneck_values})) ]
            print('Result: %s' % result)
            
        else:
            print('No checkpoint file found')
            return

if __name__ == '__main__':
    main()