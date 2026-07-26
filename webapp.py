import pathlib
import platform
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

import os
import torch
from PIL import Image
from flask import Flask, render_template, request, redirect

# โหลดโมเดล YOLOv5 (ปรับค่าความมั่นใจเป็น 0.75 เพื่อกรองตัวที่ไม่ใช่ปูออก)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', trust_repo=True)
model.conf = 0.75

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return
            
        img = Image.open(file.stream)
        results = model(img)
        
        os.makedirs("static", exist_ok=True)
        results.render()
        for im in results.ims:
            im_base64 = Image.fromarray(im)
            im_base64.save("static/image0.jpg")

        return render_template("index.html", result_image="static/image0.jpg")

    return render_template("index.html", result_image=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)