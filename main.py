import tensorflow as tf
import numpy as np
from ops import *
from utils import *
from glob import glob
import os
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import queue

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

    def load(self):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(os.getcwd()+"/training/"))

    def test(self):
        for path in ["temp2.png"]:
        # for path in sorted(glob(os.path.join("./imgs", "*.jpg")))[:10]:
            im = np.expand_dims(get_image_fit(path), 0)
            open_cv_image = np.array(im[0])

            gen = self.sess.run([self.generated], feed_dict={self.images: im})[0]
            gen = gen[0,:,:,0]
            scipy.misc.imsave("temp.jpg", gen)

            img = cv2.imread('temp.jpg')
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            zones = np.zeros_like(img)
            groupnum = 1
            boxes = []

            v = []
            for x in range(gen.shape[0]):
                for y in range(gen.shape[1]):
                    v.append((gray[x,y], x, y))

            v.sort(key=lambda x: x[0])
            v = v[::-1]
            for g, x, y in v:
                # print("%d %d %d" % (x, y, gray[x, y]))
                if gray[x,y] >= 150:
                    q = queue.Queue()
                    q.put((x,y, int(gray[x,y]), -1, -1))
                    minx = -1
                    miny = -1
                    maxx = -1
                    maxy = -1

                    # 200 -> 180, 50
                    # 100 -> 150 X
                    # oldval -> newval

                    while not q.empty():
                        ix, iy, oldval, oldx, oldy = q.get()
                        change = oldval - int(gray[ix, iy])
                        # print("%d %d" % (oldval, int(gray[ix, iy])))
                        # print("%d %d %d %d %d" % (ix, iy, change, oldx, oldy))
                        if gray[ix, iy] >= 45 and change > -30:
                            placeval = int(gray[ix, iy])
                            gray[ix, iy] = 0
                            zones[ix, iy] = groupnum * 20
                            if ix > 0:
                                q.put((ix-1, iy, placeval, ix, iy))
                            if ix < gen.shape[0] - 1:
                                q.put((ix+1, iy, placeval, ix, iy))
                            if iy > 0:
                                q.put((ix, iy-1, placeval, ix, iy))
                            if iy < gen.shape[1] - 1:
                                q.put((ix, iy+1, placeval, ix, iy))

                            if minx == -1 or ix < minx:
                                minx = ix
                            if miny == -1 or iy < miny:
                                miny = iy
                            if maxx == -1 or ix > maxx:
                                maxx = ix
                            if maxy == -1 or iy > maxy:
                                maxy = iy
                    # print("%d %d %d %d" % (minx, miny, maxx, maxy))
                    # print("%d %d %d %d" % (minx*32, miny*32, maxx*32, maxy*32))
                    cv2.rectangle(open_cv_image,(miny*32,minx*32),(maxy*32 + 32,maxx*32 + 32),(0,1,0),2)
                    boxes.append((miny*32, minx*32, maxy*32 + 32, maxx*32 + 32))
                    groupnum += 1

            return boxes

            # fig = plt.figure(figsize=(10,10))
            # a = fig.add_subplot(2,2,1)
            # a.imshow(1 - gen, cmap="Greys", interpolation="nearest")
            # a = fig.add_subplot(2,2,2)
            # a.imshow(zones, cmap="Greys", interpolation="nearest")
            # a = fig.add_subplot(2,2,3)
            # a.imshow(open_cv_image, interpolation="nearest")
            # plt.show()



# model = Inpaint()
# # model.train()
# model.load()
# model.test()
