# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Code for training the prediction model."""

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from prediction_input import build_tfrecord_input
from prediction_model import construct_model
from prediction_train import Model, FLAGS

def main(unused_argv):

    print 'Constructing models and inputs.'

    FLAGS.train_val_split = 0.
    # turn off scheduled sampling
    FLAGS.schedsamp_k = -1

    with tf.variable_scope('model', reuse=None):
        val_images, val_actions, val_states = build_tfrecord_input(training=False, shuffle=False, num_epochs=1)
        val_model = Model(val_images, val_actions, val_states, FLAGS.sequence_length)

    print 'Constructing saver.'
    # Make saver.
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES), max_to_keep=0)

    # Make training session.
    with tf.Session() as sess:
        print 'restoring model from', FLAGS.pretrained_model
        saver.restore(sess, FLAGS.pretrained_model)

        tf.initialize_local_variables().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        itr = 0
        acc_cost = 0.
        saved = False
        try:
            while not coord.should_stop():
                # Run through validation set.
                feed_dict = {
                    val_model.lr: 0.0,
                    val_model.prefix: 'val',
                    val_model.iter_num: np.float32(itr),
                }
                masks, kernels, raw_kernels, tf_layers, gen_images, gt_images, cost, val_summary_str = sess.run([
                    val_model.masks,
                    val_model.kernels,
                    val_model.gen_raw_kernels,
                    val_model.tf_layers,
                    val_model.gen_images,
                    val_model.gt_images,
                    val_model.loss,
                    val_model.summ_op,
                    ], feed_dict)
                print len(raw_kernels), len(raw_kernels[5]), raw_kernels[5][0].shape
                print np.ptp(raw_kernels[5][0][:, :, 0, 3])
                if not saved:
                    print np.shape(masks)
                    print np.shape(kernels)
                    print np.shape(tf_layers)
                    print np.shape(gen_images)
                    print np.shape(gt_images)
                    np.save('l1-model/masks', masks)
                    np.save('l1-model/kernels', kernels)
                    np.save('l1-model/tf_layers', tf_layers)
                    np.save('l1-model/gen_images', gen_images)
                    np.save('l1-model/gt_images', gt_images)
                    saved = True

                acc_cost += cost * FLAGS.batch_size
                # Print info: iteration #, cost.
                print itr, cost
                itr += 1
        except tf.errors.OutOfRangeError:
            print 'epoch limit reached'
        finally:
            coord.request_stop()

        n_samples = itr * FLAGS.batch_size
        print 'test sample count', n_samples
        print 'test average loss', (acc_cost / n_samples)

if __name__ == '__main__':
  app.run()
