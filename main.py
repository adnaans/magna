import tensorflow as tf
import numpy as np
from ops import *
from utils import *
from glob import glob
import os

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
        h3 = lrelu(conv2d(h2, 128, 128, name='dg_h3_conv')) #/16
        h4 = conv2d(h2, 128, 1, name='dg_h4_conv') #/32

        self.loss = -tf.reduce_sum(self.images * tf.log(1e-8 + generated) + (1-self.images) * tf.log(1e-8 + 1 - generated),axis=[1,2,3])

        self.optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())


    def train(self):
        data = glob(os.path.join("/imgs", "*.jpg"))
        for e in xrange(50):
            for i in range((len(data) / self.batch_size) - 1):

                batch_files = data[i*self.batch_size:(i+1)*self.batch_size]
                batch = [get_image(batch_file, 32, is_crop=True) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                batch_images += 1
                batch_images /= 2
                broken_images = np.copy(batch_images)
                broken_images[:,16:16+32,16:16+32,:] = 0

                dloss, _ = self.sess.run([self.d_loss, self.d_optim], feed_dict={self.images: batch_images, self.broken_images: broken_images})
                gloss, _ = self.sess.run([self.g_loss, self.g_optim], feed_dict={self.images: batch_images, self.broken_images: broken_images})
                print "%f, %f" % (dloss, gloss)
                if (i % 30 == 0) or (gloss > 10000):
                    fill = self.sess.run(self.genfull, feed_dict={self.images: batch_images, self.broken_images: broken_images})
                    ims("results/"+str(e*10000 + i)+".jpg",merge_color(fill,[8,8]))
                    ims("results/"+str(e*10000 + i)+"-base.jpg",merge_color(fill,[8,8]))



model = Inpaint()
model.train()
