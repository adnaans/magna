from bottle import route, run, template, static_file, get, post, request, BaseRequest
import cv2
import numpy as np
import base64
import tensorflow as tf
import json

from main import *

BaseRequest.MEMFILE_MAX = 1000 * 1000

i = Inpaint()
i.load()

@route('/pic', method='GET')
def do_uploadc():
    # lines = request.files.get('lines')
    # colors = request.files.get('colors')
    bounds = i.test()
    return json.dumps(bounds)

run(host="0.0.0.0", port=8000)
