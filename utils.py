import scipy.misc
import numpy as np
import random
import tensorflow as tf


def get_image(image_path):
    im = imread(image_path)
    resize = scipy.misc.imresize(im, 512/im.shape[0])
    if resize.shape[1] < 512:
        res = np.ones((512,512, 3))
        res[:resize.shape[0], :resize.shape[1], :] = resize
    elif resize.shape[1] >= 512:
        res = resize[:512,:512]
    res = scipy.misc.imresize(res, (512,512))
    res = transform(res)
    return res

def get_image_class(image_path):
    im = imread(image_path)
    #print(im.shape)
    if im.shape[0] > 80 or im.shape[1] > 80:
        res = np.zeros((16,16))
    else:
        resize = scipy.misc.imresize(im, 16/im.shape[0], interp="nearest")
        if resize.shape[1] < 16:
            res = np.zeros((16,16))
            res[:resize.shape[0], :resize.shape[1]] = resize[:,:,0]
        elif resize.shape[1] >= 16:
            res = resize[:16, :16, 0]
    res = scipy.misc.imresize(res, (16,16))
    res = transform(res)
    return np.expand_dims(res, 3)

def transform(image):
    return np.array(image)/255.0

def center_crop(x, crop_h, crop_w=None, resize_w=108):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return x[j:j+crop_h, i:i+crop_w]

def imread(path):
    readimage = scipy.misc.imread(path,mode="RGB").astype(np.float)
    return readimage

def merge_color(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def ims(name, img):
    # print img[:10][:10]
    scipy.misc.toimage(img, cmin=0, cmax=1).save(name)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
