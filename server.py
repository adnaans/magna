from bottle import route, run, get, post, request
from PIL import Image
import base64
import cv2
import numpy as np
import textwrap
from crop import translate

@get('/pstimg') # or @route('/login')
def login():
    return '''
        <form action="/pstimg" method="post" enctype="multipart/form-data">
            <input type="file" name="scan" accept="image/*">
            <input type="submit">
        </form>
    '''

@post('/pstimg') # or @route('/login', method='POST')
def do_postimg():
    scan = request.files.get('scan')
    # #print scan
    # # img = cv2.imdecode(np.fromstring(scan.file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    # # scan = '<img src="data:image/png;base64,{0}">'.format(scan)
    # #img = img.rotate(180)
    # # print img
    # # for i in range(100,200):
    # #     for j in range(100, 200):
    # #         img[i][j][0]=255
    # #         img[i][j][1]=255
    # #         img[i][j][2]=255

    # # cutChar(img, [[(x1,y1),(x2,y2)]])

    # x1=545
    # y1=600
    # x2=600
    # y2=775

    # img = cutChar(img, [[(x1,y1),(x2,y2)]])

    # lines = textwrap.wrap("smae smae same same smea eefj s", width=4)
    # font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    # startx = x1
    # starty = y1 + 20
    # for line in lines:
    #     cv2.putText(img,line,(startx,starty ), font, 1,(0,0,0),1, cv2.LINE_AA,False)
    #     starty+=25

    # scan = cv2.imencode(".png",img)[1]
    bsixfour = base64.b64encode(translate(scan.file.read()))
    #print bsixfour
    return '<img src="data:image/png;base64, '+bsixfour+'"/>'
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