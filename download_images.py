import json
import numpy as np
import cv2
import requests
import xml.etree.ElementTree as ET
import urllib
from PIL import Image
import io
import scipy.misc

maxsize = 512

count = 0

for i in xrange(1000):
    stringreturn = requests.get("http://danbooru.donmai.us/posts.json?tags=translated&limit=20&page=%d" % i)
    posts = stringreturn.json()
    for post in posts:
        imgurl = "http://danbooru.donmai.us" + post["file_url"]
        if ("png" in imgurl) or ("jpg" in imgurl):
            count += 1
            r = requests.get(imgurl)
            i = Image.open(io.StringIO(r.content)).convert('RGB')
            open_cv_image = np.array(i)
            img = open_cv_image[:, :, ::-1].copy()
            boolimage = np.zeros((img.shape[0]/32, img.shape[1]/32, 3))

            notesall = requests.get("http://danbooru.donmai.us/notes.json?group_by=note&search[post_id]=%s" % post["id"]).json()
            for notes in notesall:
                x = int(notes["x"])
                y = int(notes["y"])
                w = int(notes["width"])
                h = int(notes["height"])
                # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                boolimage[(y/32):((y+h)/32)+1, (x/32):((x+w)/32)+1, :] = 255;

            # scipy.misc.imsave("imgs/"+str(count)+"-b.jpg", boolimage)
            cv2.imwrite("imgs/"+str(count)+".jpg", img)
            cv2.imwrite("imgs-classes/"+str(count)+"-b.jpg", boolimage)
