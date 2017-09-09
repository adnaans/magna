import json
import numpy as np
import requests
import xml.etree.ElementTree as ET
import urllib
from PIL import Image
import io
import scipy.misc
import sys

maxsize = 512

count = int(sys.argv[1])*200

for i in range(1000):
    stringreturn = requests.get("http://danbooru.donmai.us/posts.json?tags=translated&limit=200&page=%d" % (i+int(sys.argv[1])))
    posts = stringreturn.json()
    for post in posts:
        if "file_url" in post:
            imgurl = "http://danbooru.donmai.us" + post["file_url"]
            if ("png" in imgurl) or ("jpg" in imgurl):
                count += 1
                r = requests.get(imgurl)
                i = Image.open(io.BytesIO(r.content)).convert('RGB')
                open_cv_image = np.array(i)
                img = scipy.misc.imresize(open_cv_image, 0.5)
                boolimage = np.zeros((int(img.shape[0]/16), int(img.shape[1]/16), 3))
                print("%d %s" % (count, imgurl))
                notesall = requests.get("http://danbooru.donmai.us/notes.json?group_by=note&search[post_id]=%s" % post["id"]).json()
                for notes in notesall:
                    x = int(notes["x"])
                    y = int(notes["y"])
                    w = int(notes["width"])
                    h = int(notes["height"])
                    # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                    boolimage[(y//32):((y+h)//32)+1, (x//32):((x+w)//32)+1, :] = 255;

                scipy.misc.imsave("imgs-classes/"+str(count)+"-b.jpg", boolimage)
                scipy.misc.imsave("imgs/"+str(count)+".jpg", img)
                # cv2.imwrite("imgs/"+str(count)+".jpg", img)
                # cv2.imwrite("imgs-classes/"+str(count)+"-b.jpg", boolimage)
