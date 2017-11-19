
# -*- coding:utf-8 -*-

import os.path
import tensorflow as tf
from tensorflow.python.platform import gfile
import final_inference
import data_processing

# Disable tensorflow debugging log
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BOTTLENECK_TENSOR_SIZE = 2048
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

MODEL_DIR = '../../data/inception_dec_2015'
MODEL_FILE= 'tensorflow_inception_graph.pb'

MODEL_SAVE_PATH="../../data/transfer_learning_model"
MODEL_NAME="transfer_learning_model"

VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

# 神经网络参数的设置
LEARNING_RATE_BASE = 0.5
LEARNING_RATE_DECAY_FREQUENCY = 100
LEARNING_RATE_DECAY = 0.95
REGULARIZATION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
VALIDATION_FREQUENCY = 50 
STEPS = 4000
BATCH = 50 

# 输入以下指令查看日志
# tensorboard --logdir=../../data/log/supervisor.log
SUMMARY_DIR = "../../data/log/supervisor.log"


def main():
    image_lists = data_processing.create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())
    
    # 读取已经训练好的Inception-v3模型。
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
        graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
        ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')

    with tf.name_scope('intermediate_variables'):
        global_step = tf.Variable(0, trainable=False)
        logits = final_inference.inference(bottleneck_input, BOTTLENECK_TENSOR_SIZE, n_classes)
        final_tensor = tf.nn.softmax(logits)
    
    with tf.name_scope('moving_average'):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.name_scope('loss_function'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean

    with tf.name_scope('final_training_ops'):
        learning_rate = tf.train.exponential_decay(
                LEARNING_RATE_BASE,
                global_step,
                LEARNING_RATE_DECAY_FREQUENCY,
                LEARNING_RATE_DECAY,
                staircase=True)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 计算正确率。
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('accuracy', evaluation_step)

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        # 生成日志对象
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        # 训练过程。
	print 'Start trainning'
        for i in range(STEPS):
            train_bottlenecks, train_ground_truth = data_processing.get_random_cached_bottlenecks(
                sess, n_classes, image_lists, BATCH, 'training', jpeg_data_tensor, bottleneck_tensor)
            # 运行训练步骤以及所有的日志生成操作，得到这次运行的日志
            summary, _, cur_step = sess.run([merged, train_op, global_step], feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})

            if cur_step % VALIDATION_FREQUENCY == 0:
                validation_bottlenecks, validation_ground_truth = data_processing.get_random_cached_bottlenecks(
                    sess, n_classes, image_lists, BATCH, 'validation', jpeg_data_tensor, bottleneck_tensor)
                validation_accuracy = sess.run(evaluation_step, feed_dict={
                    bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                print('Step\t%d: Validation accuracy on random sampled %d examples = %.1f%%' %
                    (cur_step, BATCH, validation_accuracy * 100))
                # 初始化TensorFlow持久化类。
                saver = tf.train.Saver()
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))

            # 将得到的所有日志写入日志文件，这样TensorBoard程序就可以拿到这次运行所对应的
            # 运行信息。
            summary_writer.add_summary(summary, i)
            
        # 在最后的测试数据上测试正确率。
        # test_bottlenecks, test_ground_truth = get_test_bottlenecks(
        #     sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor)
        test_bottlenecks, test_ground_truth = data_processing.get_random_cached_bottlenecks(
            sess, n_classes, image_lists, BATCH, 'testing', jpeg_data_tensor, bottleneck_tensor)
        test_accuracy = sess.run(evaluation_step, feed_dict={
            bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

    summary_writer.close()

if __name__ == '__main__':
    main()
