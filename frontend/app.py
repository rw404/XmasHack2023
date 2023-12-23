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

app = Flask(__name__)


@lru_cache
def get_client():
    return InferenceServerClient(url="localhost:8500")


def main_back(img_list):
    triton_client = get_client()

    first_img = img_list[0]
    second_img = img_list[1]
    third_img = img_list[2]
    fourth_img = img_list[3]

    print(first_img.shape)
    print(second_img.shape)
    print(third_img.shape)
    print(fourth_img.shape)

    inputs = []
    outputs = []
    inputs.append(InferInput("input", [1, 3, 64, 64], "FP32"))
    inputs[0].set_data_from_numpy(first_img)

    inputs.append(InferInput("onnx::Add_1", [1, 3, 128, 128], "FP32"))
    inputs[1].set_data_from_numpy(second_img)

    inputs.append(InferInput("onnx::Add_2", [1, 3, 256, 256], "FP32"))
    inputs[2].set_data_from_numpy(third_img)

    inputs.append(InferInput("onnx::Add_3", [1, 3, 512, 512], "FP32"))
    inputs[3].set_data_from_numpy(fourth_img)

    outputs.append(InferRequestedOutput("output"))
    outputs.append(InferRequestedOutput("onnx::Add_872"))
    outputs.append(InferRequestedOutput("onnx::Add_1205"))
    outputs.append(InferRequestedOutput("1467"))

    results = triton_client.infer("onnx-mnist", inputs, outputs=outputs)

    answer = []
    answer.append(np.squeeze(results.as_numpy("output")))
    answer.append(np.squeeze(results.as_numpy("onnx::Add_872")))
    answer.append(np.squeeze(results.as_numpy("onnx::Add_1205")))
    answer.append(np.squeeze(results.as_numpy("1467")))

    return answer


def decomposition(img, level_num=4):
    G_list = []
    L_list = []
    G_list.append(img)
    for i in range(level_num - 1):
        G_list.append(cv2.pyrDown(G_list[i]))
    for j in range(level_num - 1):
        L_list.append(
            G_list[j]
            - cv2.pyrUp(G_list[j + 1], dstsize=(G_list[j].shape[1], G_list[j].shape[0]))
        )
    L_list.append(G_list[level_num - 1])
    G_list.reverse()
    L_list.reverse()
    return G_list, L_list


# from model import MSPEC_Net

IN_SIZE = 512


def img_to_list(img):
    maxsize = max([img.shape[0], img.shape[1]])
    scale_ratio = IN_SIZE / maxsize
    im_low = cv2.resize(
        img, (0, 0), fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_CUBIC
    )
    top_pad, left_pad = IN_SIZE - im_low.shape[0], IN_SIZE - im_low.shape[1]
    img = cv2.copyMakeBorder(im_low, top_pad, 0, left_pad, 0, cv2.BORDER_DEFAULT)

    if img.dtype == "uint8":
        img = img / 255
    _, L_list = decomposition(img)

    L_list = [
        torch.from_numpy(data)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .detach()
        .cpu()
        .numpy()
        for data in L_list
    ]

    return L_list, (top_pad, left_pad)


def list_to_img(Y_list, pad, shape):
    out = torch.from_numpy(Y_list[-1]).squeeze().permute(1, 2, 0).detach().cpu().numpy()
    out = out[pad[0] :, pad[1] :, :]

    out = cv2.resize(out, (shape[1], shape[0])) * 255
    out = out.clip(0, 255).astype(np.uint8)

    return out


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
            orig_shape = img.shape
            lisst, pad = img_to_list(img)
            # ------- debug
            print(lisst.__len__())
            print(pad)
            # -------
            result = main_back(lisst)
            res_img = list_to_img(result, pad, orig_shape)
            # cv2.imwrite("tests/ans.png", out)
            if orig_shape[0] > orig_shape[1]:
                res_img = np.hstack([old_img, res_img])
            else:
                res_img = np.vstack([old_img, res_img])
            cv2.imwrite("static/converted_img.jpg", res_img)
            # Image.fromarray(res_img).save("static/converted_img.jpg")
            data["converted_img"] = "static/converted_img.jpg"

        print(data)
    return render_template("index.html", flask_data=data)


if __name__ == "__main__":
    app.run(debug=True)