import base64
import cgi
import io
import urllib

import numpy as np
from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__)
app.config['UPLOADED_FILES'] = 'static/uploads'

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
            img = np.array(Image.open('color_image.jpg'))
            res_img = np.hstack([img, img])
            Image.fromarray(res_img).save('static/converted_color_img.jpg')
        elif 'convert_mask_img' in request.form:
            img = np.array(Image.open('mask_image.jpg'))
            res_img = np.hstack([img, img])
            Image.fromarray(res_img).save('static/converted_mask_img.jpg')
        else:
            for i, file_storage in enumerate(request.files.getlist('files[]')):
                Image.open(io.BytesIO(file_storage.read())).save(f'static/uploads/image_{i}.jpg')

    return render_template('index.html')


def create_img_gallery(old_imgs, new_imgs):
    pass

if __name__ == '__main__':
    app.run(debug=True)
