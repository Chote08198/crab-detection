import pathlib
temp = pathlib.PosixPath
pathlib.WindowsPath = pathlib.PosixPath

import pathlib
pathlib.PosixPath = pathlib.WindowsPath

from flask import Flask, render_template, request, redirect
import torch
import os
from PIL import Image

app = Flask(__name__)

from ultralytics import YOLO
model = YOLO('best.pt')


# ----------------------------------------------------
# ⚙️ ปรับตั้งค่าความแม่นยำในการจับกรอบตรงนี้:
model.conf = 0.60  # แสดงเฉพาะวัตถุที่มั่นใจมากกว่า 60% ขึ้นไป (ช่วยลดกรอบมั่ว)
model.iou = 0.40   # ตัดกรอบที่จับซ้อนทับตำแหน่งเดียวกันออก (ช่วยลดกรอบซ้ำ)
# ----------------------------------------------------

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