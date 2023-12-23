import base64
import io

import cv2
import numpy as np
from flask import Flask, render_template, request
from PIL import Image
from tritonclient.http import (InferenceServerClient, InferInput,
                               InferRequestedOutput)

from functools import lru_cache

import time
import os
import cv2
from gevent import monkey
monkey.patch_all()

app = Flask(__name__)
app.config['UPLOADED_FILES'] = 'static/uploads'

@lru_cache
def get_client():
    return InferenceServerClient(url="0.0.0.0:8500")

def get_correction_mask(orig, proc, quantil=20):
    orig = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY).astype(float)
    proc = cv2.cvtColor(proc, cv2.COLOR_RGB2GRAY).astype(float)

    orig = orig - orig.min()
    orig = orig / orig.max() * 255

    proc = proc - proc.min()
    proc = proc / proc.max() * 255

    mask = abs(orig - proc)
    fmask = mask.flatten()
    _, th = np.histogram(fmask, bins=100)

    mask[mask < th[quantil]] = 0
    mask[mask >= th[quantil]] = 255

    sig = np.sign(orig - proc)
    flash = ([sig > 0] * mask)[0].astype(np.uint8)
    dark = ([sig < 0] * mask)[0].astype(np.uint8)
    z = np.zeros_like(sig, dtype=np.uint8)

    return np.stack([flash, z, dark], axis=-1).astype(np.uint8)

def main_back(img_list):
    triton_client = get_client()
    orig = img_list.copy()
    
    first_img = img_list

    print(first_img.shape)

    inputs = []
    outputs = []
    inputs.append(InferInput("input", [512, 512, 3], "UINT8"))
    inputs[0].set_data_from_numpy(first_img)
    
    
    outputs.append(InferRequestedOutput("output"))
    
    results = triton_client.infer("python-unet", inputs, outputs=outputs)

    answer = []
    answer.append(np.squeeze(results.as_numpy("output")))
    
    res = answer[0].clip(0, 255).astype(np.uint8)
    
    return res

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route('/', methods=['GET', 'POST'])
def main():
    #data = {'converted_img': None}
    if request.method == 'POST':
        if 'color_img' in request.form:
            img_data = base64.b64decode(request.form['color_img'].split(',')[1])
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
            img.save('color_image.jpg')
            
            
            
        elif 'mask_img' in request.form:
            img_data = base64.b64decode(request.form['mask_img'].split(',')[1])
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
            img.save('mask_image.jpg')
            
        elif 'convert_color_img' in request.form:
            img = cv2.imread("color_image.jpg")
        
            old_img = img.copy()
            old_img = cv2.cvtColor(old_img, cv2.COLOR_BGR2RGB)
            
            inp_size = img.shape[:2]

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (512, 512))
            
            result = main_back(img)
            res_img = cv2.resize(result, inp_size[::-1])
            
            print(res_img.shape)
            print(old_img.shape)
            res_img = np.hstack([old_img, res_img])
            res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite("static/converted_color_img.jpg", res_img)
            
        elif 'convert_mask_img' in request.form:
            img = cv2.imread("mask_image.jpg")
            old_img = img.copy()
            old_img = cv2.cvtColor(old_img, cv2.COLOR_BGR2RGB)
            
            inp_size = img.shape[:2]

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (512, 512))
            
            result = main_back(img)
            res_img = cv2.resize(result, inp_size[::-1])
            res_img = get_correction_mask(old_img, res_img.copy())# MASKS
            
            res_img = np.hstack([old_img, res_img])
            res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)
            
            print("HUI")
            cv2.imwrite("static/converted_mask_img.jpg", res_img)
        else:
            imgs = []
            for i, file_storage in enumerate(request.files.getlist('files[]')):
                img = Image.open(io.BytesIO(file_storage.read())).convert('RGB')
                img.save(f'static/uploads/image_{i}.jpg')
                img = np.array(img)
                imgs.append(img)
                
            # convert imgs
            gallery_img = create_img_gallery(imgs, imgs)
            Image.fromarray(gallery_img).save('static/gallery_img.jpg')

    return render_template('index.html')


def create_img_gallery(old_imgs, new_imgs):
    width = old_imgs[0].shape[1]
    res_img = []
    for old_img, new_img in zip(old_imgs, new_imgs):
        h, w = old_img.shape[:2]
        ratio = width / w
        new_h = int(ratio * h)
        new_w = int(ratio * w) 
        old_img = cv2.resize(old_img, (new_w, new_h))
        new_img = cv2.resize(new_img, (new_w, new_h))
        res_img.append(np.hstack([old_img, new_img]))
    return np.vstack(res_img)

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
