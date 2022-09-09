from flask import Flask, render_template , request , jsonify
from PIL import Image
import sys
import os
import io
import numpy as np
import cv2
import base64
from app.detect import *


app = Flask(__name__)


@app.route('/detectObject',methods=['POST'])
def mask_image():
        file = request.files['image'].read()
        npimg= np.fromstring(file,np.uint8)

        img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)

        my_model = Licence_Plate(trained_model)

        img = my_model.detect(img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img.astype("uint8"))
        rawBytes = io.BytesIO()
        img.save(rawBytes, "JPEG")

        rawBytes.seek(0)

        img_base64 = base64.b64encode(rawBytes.read())

        return jsonify({'status':str(img_base64)})

@app.route('/test' , methods=['GET','POST'])
def test():
    print("log: got at test" , file=sys.stderr)
    return jsonify({'status':'succces'})

@app.route('/')
def home():
    return render_template('./index.html')


@app.after_request
def after_request(response):
    print("log: setting cors" , file = sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response
