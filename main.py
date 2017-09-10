import tensorflow as tf
import numpy as np
from ops import *
from utils import *
from glob import glob
import os
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Inpaint():
    def __init__(self):
        self.num_colors = 3
        self.batch_size = 64

        self.images = tf.placeholder(tf.float32, [None, None, None, self.num_colors])
        self.classes = tf.placeholder(tf.float32, [None, None, None, 1])

        # breaking down the context
        h0 = lrelu(conv2d(self.images, self.num_colors, 64, name='dg_h0_conv')) #/2
        h1 = lrelu(conv2d(h0, 64, 64, name='dg_h1_conv')) #/4
        h2 = lrelu(conv2d(h1, 64, 64, name='dg_h2_conv')) #/8
        h3 = lrelu(conv2d(h2, 64, 128, name='dg_h3_conv')) #/16
        h4 = tf.sigmoid(conv2d(h3, 128, 1, name='dg_h4_conv')) #/32
        generated = h4
        self.generated = generated
        self.loss = -tf.reduce_sum(self.classes * tf.log(1e-8 + generated) + (1-self.classes) * tf.log(1e-8 + 1 - generated),axis=[1,2,3])

        self.optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

        self.saver = tf.train.Saver(max_to_keep=10)


    def train(self):
        data = sorted(glob(os.path.join("./imgs", "*.jpg")))
        data_real = sorted(glob(os.path.join("./imgs-classes", "*.jpg")))
        for e in range(50):
            for j in range((len(data) // self.batch_size) - 1):
                i = j + 0
                batch_files = data[i*self.batch_size:(i+1)*self.batch_size]
                batch = [get_image(batch_file) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)

                class_files = data_real[i*self.batch_size:(i+1)*self.batch_size]
                classes = [get_image_class(cfile) for cfile in class_files]
                class_images = np.array(classes).astype(np.float32)

                loss, gen, _ = self.sess.run([self.loss, self.generated, self.optim], feed_dict={self.images: batch_images, self.classes: class_images})
                print("%d %d, %f" % (e, i, np.mean(loss)))
                if i % 30 == 0:
                    self.saver.save(self.sess, os.getcwd()+"/training/train",global_step=e*1000000 + i)

    def test(self):
        data = sorted(glob(os.path.join("./imgs", "*.jpg")))
        data_real = sorted(glob(os.path.join("./imgs-classes", "*.jpg")))

        self.saver.restore(self.sess, tf.train.latest_checkpoint(os.getcwd()+"/training/"))

        num = 14

        batch_files = data[0:num]
        batch = [get_image(batch_file) for batch_file in batch_files]
        batch_images = np.array(batch).astype(np.float32)

        class_files = data_real[0:num]
        classes = [get_image_class(cfile) for cfile in class_files]
        class_images = np.array(classes).astype(np.float32)

        loss, gen = self.sess.run([self.loss, self.generated], feed_dict={self.images: batch_images, self.classes: class_images})
        print(loss)
        for i in range(num):
            scipy.misc.imsave("res/" + str(i)+".jpg", gen[i,:,:,0])
            
            fig = plt.figure()
            a = fig.add_subplot(1,3,1)
            a.imshow(1 - gen[i,:,:,0], cmap="Greys", interpolation="nearest")
            a = fig.add_subplot(1,3,2)
            a.imshow(1 - class_images[i,:,:,0], cmap="Greys", interpolation="nearest")
            a = fig.add_subplot(1,3,3)
            a.imshow(batch_images[i,:,:,:], interpolation="nearest")
            plt.show()


model = Inpaint()
# model.train()
model.test()
