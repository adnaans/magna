from bottle import route, run, get, post, request
from PIL import Image
import base64
import cv2
import numpy as np

@get('/pstimg') # or @route('/login')
def login():
    return '''
        <form action="/pstimg" method="post" enctype="multipart/form-data">
            <input type="file" name="scan" accept="image/*">
            <input type="submit">
        </form>
    '''

@post('/pstimg') # or @route('/login', method='POST')
def do_login():
    scan = request.files.get('scan')
    #print scan
    img = cv2.imdecode(np.fromstring(scan.file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    # scan = '<img src="data:image/png;base64,{0}">'.format(scan)
    #img = img.rotate(180)
    scan = cv2.imencode(".png",img)[1]
    bsixfour =  base64.b64encode(scan)
    print bsixfour
    return '<img src="data:image/png;base64, '+bsixfour+'"/>'
    # print img_tag
    # #return rotatedscan;
    # return img_tag


run(host='localhost', port=8080, debug=True)