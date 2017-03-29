import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

import os

from scipy.ndimage import imread

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

def cnn_model_fn(features, labels, mode):
	input_layer = tf.reshape(features, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=[5,5],
		padding="same",
		activation=tf.nn.relu
	)
	pool1 = tf.layers.max_pooling2d(
		inputs=conv1,
		pool_size=[2,2],
		strides=2
	)
	
	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=64,
		kernel_size=[5,5],
		padding="same",
		activation=tf.nn.relu
	)
	pool2 = tf.layers.max_pooling2d(
		inputs=conv2,
		pool_size=[2,2],
		strides=2
	)
	
	pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
	
	dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
	dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
	
	predictions = tf.layers.dense(inputs=dropout, units=10)
	
	loss = None
	train_op = None
	
	if mode != learn.ModeKeys.INFER:
		loss = tf.losses.absolute_difference(labels, predictions)
		# more like absolutely terrible!
		
	if mode == learn.ModeKeys.TRAIN:
		train_op = tf.contrib.layers.optimize_loss(
			loss=loss,
			global_step=tf.contrib.framework.get_global_step(),
			learning_rate=0.001,
			optimizer="SGD"
		)
		
	predictions = {
		"depths":predictions,
	}
	
	return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)
	
def load_kitti_dataset():
	dir = "KITTI-data"
	depth_dir = os.path.join(dir, "depth")
	depth_list = os.listdir(depth_dir)
	for item in depth_list:
		
	
def main(unused_argv):
# 	mnist = learn.datasets.load_dataset("mnist")
# 	train_data = mnist.train.images
# 	train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
# 	eval_data = mnist.test.images
# 	eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
	
	mnist_classifier = learn.Estimator(model_fn=cnn_model_fn, model_dir="model/")
	
	mnist_classifier.fit(
		x=train_data,
		y=train_labels,
		batch_size=100,
		steps=20000,
		monitors=[logging_hook]
	)
	
	metrics = {
		 "loss":learn.metric_spec.MetricSpec(metric_fn=tf.losses.absolute_difference, prediction_key="depths")
	}
	
	eval_results = mnist_classifier.evaluate(x=eval_data, y=eval_labels, metrics=metrics)
	print(eval_results)
	
if __name__ == "__main__":
	tf.app.run()