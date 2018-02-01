from __future__ import print_function

import argparse
import os
import sys
import time
import scipy.io as sio
from PIL import Image
import cv2
import tensorflow as tf
import numpy as np

from model import DeepLabResNetModel

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

NUM_CLASSES = 27
SAVE_DIR = './output/'
RESTORE_PATH = ''
matfn = 'color150.mat'

def get_arguments():
    parser = argparse.ArgumentParser(description="Indoor segmentation parser.")
    parser.add_argument("--restore_from", type=str, default=RESTORE_PATH,
                        help="checkpoint location")

    return parser.parse_args()

def read_labelcolours(matfn):
    mat = sio.loadmat(matfn)
    color_table = mat['colors']
    shape = color_table.shape
    color_list = [tuple(color_table[i]) for i in range(shape[0])]

    return color_list

def decode_labels(mask, num_images=1, num_classes=150):
    label_colours = read_labelcolours(matfn)

    n, h, w, c = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :, 0]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return outputs

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


class Tester(object):
    def __init__(self):
        self.name = 'Tester'
        self.session = None
        self.loader = None
        self.ckpt = None
        self.net = None
        self.in_progress = False

    def run(self):
        args = get_arguments()

        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            img = cv2.resize(frame.copy(), (300, 300))
            if ret and (not self.in_progress):

                img = tf.cast(img, dtype=tf.float32)
                self.net = DeepLabResNetModel({'data': tf.expand_dims(img, dim=0)}, is_training=False, num_classes=NUM_CLASSES)
                restore_var = tf.global_variables()
                # Predictions.
                raw_output = self.net.layers['fc_out']
                raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2, ])
                raw_output_up = tf.argmax(raw_output_up, dimension=3)
                pred = tf.expand_dims(raw_output_up, dim=3)

                # Set up TF session and initialize variables.
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                if not self.session:
                    self.session = tf.Session(config=config)
                    init = tf.global_variables_initializer()

                    self.session.run(init)
                # Load weights.

                if self.ckpt is None:
                    self.ckpt = tf.train.get_checkpoint_state(args.restore_from)
                    if self.ckpt and self.ckpt.model_checkpoint_path and (self.loader is None):
                        self.loader = tf.train.Saver(var_list=restore_var)
                        load(self.loader, self.session, self.ckpt.model_checkpoint_path)
                    else:
                        print('No checkpoint file found. or loaded!')
                print('[In Progress] Predicting')
                self.in_progress = True
                preds = self.session.run(pred)
                self.in_progress = False
                print('[Done] Prediction done!')
                msk = decode_labels(preds, num_classes=NUM_CLASSES)
                cv2.imshow('mask', msk[0])
            else:
                print('Drop frame = %s | In Progress = %s' % (not ret, self.in_progress))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


def main():
    test = Tester()
    test.run()


if __name__ == '__main__':
    main()
