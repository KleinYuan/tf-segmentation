from __future__ import print_function
from PIL import Image
from model import DeepLabResNetModel, PSPNet101, PSPNet50
import time
import cv2
import numpy as np
import scipy.io as sio
import tensorflow as tf

COLOR_MAT_FP = 'color150.mat'


class SegApp(object):
	def __init__(self, model_name='DeepLab'):
		self.name = 'App'
		self.model_name = model_name
		self.model_path = './model/{}/'.format(model_name.lower())
		self.session = None
		self.saver = None
		self.ckpt = None
		self.net = None
		self.img_tf = None
		self.cap = None
		self.prediction = None
		self.in_progress = False
		self.model_map = {
			'DeepLab':
				{'model': DeepLabResNetModel,
				 'img_shape': 300,
				 'class_num': 27
				 },
			'PSPNet101':
				{'model': PSPNet101,
				 'img_shape': 720,
				 'class_num': 19
				 },
			'PSPNet50':
				{'model': PSPNet50,
				 'img_shape': 473,
				 'class_num': 150
				 }
		}
		self.num_classes = self.model_map[model_name]['class_num']
		self.img_shape = self.model_map[model_name]['img_shape']
		self.model = self.model_map[model_name]['model']
		self.input_feed_shape = (1, self.img_shape, self.img_shape, 3)

	@staticmethod
	def decode_label_colours(mat_fp):
		mat = sio.loadmat(mat_fp)
		color_table = mat['colors']
		shape = color_table.shape
		color_list = [tuple(color_table[i]) for i in range(shape[0])]
		return color_list

	def decode_labels(self, mask, num_images=1):
		label_colours = self.decode_label_colours(COLOR_MAT_FP)
		n, h, w, c = mask.shape
		assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
			n, num_images)
		outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
		for i in range(num_images):
			img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
			pixels = img.load()
			for j_, j in enumerate(mask[i, :, :, 0]):
				for k_, k in enumerate(j):
					if k < self.num_classes:
						pixels[k_, j_] = label_colours[k]
			outputs[i] = np.array(img)
		return outputs

	def _tf_init(self):
		self.img_tf = tf.placeholder(dtype=tf.float32, shape=self.input_feed_shape)
		self.net = self.model({'data': self.img_tf}, is_training=False, num_classes=self.num_classes)

		self.config = tf.ConfigProto()
		self.config.gpu_options.allow_growth = True
		self.session = tf.Session(config=self.config)
		init = tf.global_variables_initializer()
		self.session.run(init)

		# Load pre-trained model
		self.ckpt = tf.train.get_checkpoint_state(self.model_path)
		if self.ckpt and self.ckpt.model_checkpoint_path and (self.saver is None):
			self.saver = tf.train.Saver(var_list=tf.global_variables())
			self.saver.restore(self.session, self.ckpt.model_checkpoint_path)
			print("Restored model parameters from {}".format(self.ckpt.model_checkpoint_path))
		else:
			print('No checkpoint file found. or loaded!')

	def _pre_process(self, frame):
		print('Pre-processing image ...')
		img_resized = cv2.resize(frame.copy(), (self.img_shape, self.img_shape))
		img_feed = np.array(img_resized, dtype=float)
		img_feed = np.expand_dims(img_feed, axis=0)
		print('Pre-processed image!')
		return img_resized, img_feed

	def tf_release(self):
		self.session.close()
		del self.session

	def process(self, img):
		img_resized, img_feed = self._pre_process(img)
		try:
			raw_output = self.net.layers['fc_out']
		except:
			raw_output = self.net.layers['conv6']

		raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img_resized)[0:2, ])
		raw_output_up = tf.argmax(raw_output_up, dimension=3)
		pred = tf.expand_dims(raw_output_up, dim=3)
		self.in_progress = True
		start_time = time.time()
		_ops = self.session.run(pred, feed_dict={self.img_tf: img_feed})
		self.prediction = _ops

		elapsed_time = time.time() - start_time
		print("FPS: ", 1 / elapsed_time)
		self.in_progress = False
		msk = self.decode_labels(_ops)
		over_layed = cv2.addWeighted(img_resized, 0.5, msk[0], 0.3, 0)
		return over_layed

	# def start_live_run(self):
	# 	self._tf_init()
	# 	self.cap = cv2.VideoCapture(0)
	# 	while True:
	# 		ret, frame = self.cap.read()
	# 		if ret and (not self.in_progress):
	# 			over_layed = self.process(frame)
	# 			cv2.imshow('over_layed', over_layed)
	#
	# 		if cv2.waitKey(1) & 0xFF == ord('q'):
	# 			self.tf_release()
	# 			break
	#
	# 	self.cap.release()
	# 	cv2.destroyAllWindows()
	# 	self.tf_release()

	# def demo_one_img(self, img_fp):
	# 	self._tf_init()
	# 	img = cv2.imread(img_fp)
	# 	over_layed = self.process(img)
	# 	cv2.imshow('over_layed', over_layed)
	# 	cv2.waitKey(0)
	# 	self.cap.release()
	# 	cv2.destroyAllWindows()
	# 	self.tf_release()

	def spin(self):
		self._tf_init()

	def get_result(self):
		return self.prediction


'''
API:
1. Initial an instance of app first : app = App()
2. Spin service: app.spin()
3. Feed img in np.array/opencv format: app.process(img=img)
4. Get prediction: app.get_result()
'''
