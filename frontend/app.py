import base64
import io
from functools import lru_cache

import cv2
import numpy as np
import torch
from flask import Flask, render_template, request
from PIL import Image
from tritonclient.http import (InferenceServerClient, InferInput,
                               InferRequestedOutput)
from gevent import monkey
monkey.patch_all()

app = Flask(__name__)


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
    # res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB) # photo
    
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


@app.route("/", methods=["GET", "POST"])
def main():
    data = {"converted_img": None}
    if request.method == "POST":
        if "img" in request.form:
            img_data = base64.b64decode(request.form["img"].split(",")[1])
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            img.save("image.jpg")
            data["converted_img"] = None
        elif "convert_img" in request.form:
            # img = np.array(Image.open("image.jpg"))
            img = cv2.imread("image.jpg")
            old_img = img.copy()
            old_img = cv2.cvtColor(old_img, cv2.COLOR_BGR2RGB)
            
            inp_size = img.shape[:2]

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (512, 512))
            #lisst, pad = img_to_list(img)
            # ------- debug
            #print(lisst.__len__())
            #print(pad)
            # -------
            result = main_back(img)
            res_img = cv2.resize(result, inp_size[::-1])
            res_img = get_correction_mask(old_img, res_img.copy())# MASKS
            
            print(res_img.shape)
            print(old_img.shape)
            res_img = np.hstack([old_img, res_img])
            res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite("static/converted_img.jpg", res_img)
            # Image.fromarray(res_img).save("static/converted_img.jpg")
            data["converted_img"] = "static/converted_img.jpg"

        print(data)
    return render_template("index.html", flask_data=data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)