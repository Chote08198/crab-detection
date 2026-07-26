import pathlib
import torch
import os
from PIL import Image
from flask import Flask, render_template, request, redirect

temp = pathlib.PosixPath
pathlib.WindowsPath = pathlib.PosixPath

# โหลดโมเดล YOLOv5 ดั้งเดิมที่รองรับ .render() และ .ims ของคุณ
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', trust_repo=True)

app = Flask(__name__)

# ตั้งค่าความแม่นยำ (ย้ายมาไว้ตรงนี้ถูกต้องแล้วครับ)
model.conf = 0.60
model.iou = 0.40

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        img = Image.open(file.stream)
        results = model(img, size=640)

        os.makedirs("static", exist_ok=True)
        results.render()
        for img_arr in results.ims:
            im_base64 = Image.fromarray(img_arr)
            im_base64.save("static/image0.jpg")

        return render_template("index.html", result_image="static/image0.jpg")

    return render_template("index.html", result_image=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)