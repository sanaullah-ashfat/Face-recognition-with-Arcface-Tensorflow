import io
import re
import os
import cv2
import json
import math
import base64
import warnings
import numpy as np
from PIL import Image
from flask_cors import CORS
from task import TASK
from main.RECOGNITION import RECOG
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
warnings.filterwarnings('ignore')
app = Flask(__name__)
CORS(app)


def convert_to_im_array(data):
    arr = base64.b64decode(data)
    img_arr = np.frombuffer(arr, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return img


@app.route('/recognition', methods=['POST'])
def get_predictions():
    
    if request.method == 'POST':
        try:
            data = json.loads(request.data.decode('utf-8'))
            file = data['file']
            img_bytes = file
            img_bytes = convert_to_im_array(img_bytes)
            img = np.asarray(img_bytes)
            # task = "recognition"
            # res =TASK()
            # result= res.response(img, task,ID= None)
            recg=RECOG()
            result=recg.recognition(img)
            return jsonify(result)
        except Exception as e:
           # print(e)
            return jsonify({'result': 'error during prediction', 'error': e})
        return jsonify(result)

if __name__ == '__main__':
    print("SERVER STARTED")
    app.run()

