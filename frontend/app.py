import base64
import cgi
import io
import urllib

import numpy as np
from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__)


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
    data = {'converted_img': None}
    if request.method == 'POST':
        if 'img' in request.form:
            img_data = base64.b64decode(request.form['img'].split(',')[1])
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
            img.save('image.jpg')
            data['converted_img'] = None
        elif 'convert_img' in request.form:
            img = np.array(Image.open('image.jpg'))
            if img.shape[0] > img.shape[1]:
                res_img = np.hstack([img, img])
            else:
                res_img = np.vstack([img, img])
            Image.fromarray(res_img).save('static/converted_img.jpg')
            data['converted_img'] = 'static/converted_img.jpg'

        print(data)
    return render_template('index.html', flask_data=data)


if __name__ == '__main__':
    app.run(debug=True)
