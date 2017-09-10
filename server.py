from bottle import route, run, get, post, request
from PIL import Image
import base64
import cv2
import numpy as np
from crop import *
import textwrap
import json


@get('/pstimg') # or @route('/login')
def login():
    return '''
        <center><h1>Welcome to Karo!</h1></center>
        <p>Upload a picture of a raw(untranslated) manga page and our servers will read, translate, and rewrite the page for you to read in English!</p>
        <form action="/pstimg" method="post" enctype="multipart/form-data">
            <input type="file" name="scan" accept="image/*">
            <input type="submit">
        </form>
    '''

@post('/pstimg') # or @route('/login', method='POST')
def do_postimg():
    scan = request.files.get('scan')
    #print scan
    # img = cv2.imdecode(np.fromstring(scan.file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    # scan = '<img src="data:image/png;base64,{0}">'.format(scan)
    #img = img.rotate(180)
    # print img
    # for i in range(100,200):
    #     for j in range(100, 200):
    #         img[i][j][0]=255
    #         img[i][j][1]=255
    #         img[i][j][2]=255

    # cutChar(img, [[(x1,y1),(x2,y2)]])

    # img = cutChar(img, [[(100,100),(300,300)]])

    # scan = cv2.imencode(".png",img)[1]
    
    bsixfour = base64.b64encode(translate(scan.file.read()))
    #print bsixfour
    return '<img src="data:image/png;base64, '+base64.b64encode(scan.file.read())+'"/>'
    # print img_tag
    # #return rotatedscan;
    # return img_tag

@post('/pstimgmob') # or @route('/login', method='POST')
def do_postimgmob():
    # print(request.body.read())
    data = json.loads(request.body.read())
    print(data['data'])
    # scan = request.files.get('scan')
    #print scan
    # img = cv2.imdecode(np.fromstring(scan.file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    # scan = '<img src="data:image/png;base64,{0}">'.format(scan)
    #img = img.rotate(180)
    # print img
    # for i in range(100,200):
    #     for j in range(100, 200):
    #         img[i][j][0]=255
    #         img[i][j][1]=255
    #         img[i][j][2]=255

    # cutChar(img, [[(x1,y1),(x2,y2)]])

    # img = cutChar(img, [[(100,100),(300,300)]])

    # scan = cv2.imencode(".png",img)[1]
    bsixfour = base64.b64encode(translate(base64.b64decode(data['data'])))
    #print bsixfour
    # return '<img src="data:image/png;base64, '+bsixfour+'"/>'
    # return base64.b64encode(base64.b64decode(data['data']))
    return bsixfour
    # print img_tag
    # #return rotatedscan;
    # return img_tag

def cutChar(img, spots):
    for corners in spots:
        for i in range(corners[0][1],corners[1][1]):
            for j in range(corners[0][0],corners[1][0]):
                img[i][j][0]=255
                img[i][j][1]=255
                img[i][j][2]=255

    return img



run(host='localhost', port=8080, debug=True)
