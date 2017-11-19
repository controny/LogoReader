
# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.python.platform import gfile
import cPickle
import os
import numpy as np
import data_processing
import transfer_learning

if __name__ == '__main__':
    with tf.Session() as sess:
        image_lists = data_processing.create_image_lists(0, 0)

        with gfile.FastGFile(os.path.join(transfer_learning.MODEL_DIR, transfer_learning.MODEL_FILE), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
                graph_def, return_elements=[transfer_learning.BOTTLENECK_TENSOR_NAME, transfer_learning.JPEG_DATA_TENSOR_NAME])

        with open(data_processing.CLASS_INDEX_FILE, 'rb') as f:
            label_names = cPickle.load(f)

        num_calculate = 0;
        category = 'training'
        for label_name in label_names:
            label_lists = image_lists[label_name]
            sub_dir = label_lists['dir']
            num_index = len(label_lists[category])
            sub_dir_path = os.path.join(data_processing.CACHE_DIR, sub_dir)
            if not os.path.exists(sub_dir_path): os.makedirs(sub_dir_path)

            for index in range(num_index):
                distorted = np.random.randint(data_processing.NUM_DISTORTED)
                bottleneck_path = data_processing.get_bottleneck_path(image_lists, label_name, index, category, distorted)

                if not os.path.exists(bottleneck_path):
                    try :
                        image_path = data_processing.get_image_path(image_lists, data_processing.INPUT_DATA, label_name, index, category)
                        image_data = gfile.FastGFile(image_path, 'rb').read()
                        bottleneck_values = data_processing.run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor, distorted)
                        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
                        with open(bottleneck_path, 'w') as bottleneck_file:
                            bottleneck_file.write(bottleneck_string)
                            num_calculate += 1
                            if num_calculate % 10 == 0:
                                print '%d bottlenecks have been calculated.' % num_calculate
                    except tf.errors.InvalidArgumentError, e:
                        pass
                    except tf.errors.FailedPreconditionError, e:
                        pass
                    except Exception, e:
                        logging.exception(e)