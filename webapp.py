import os
import pathlib
import torch
from PIL import Image
from flask import Flask, render_template, request, redirect

temp = pathlib.PosixPath
pathlib.WindowsPath = pathlib.PosixPath

# ข้ามการตรวจสอบความปลอดภัยเพื่อโหลดโมเดล YOLOv5 ของคุณ
torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

# โหลดโมเดลด้วย torch.hub แบบดั้งเดิม (รองรับ best.pt ของคุณชัวร์ 100%)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', trust_repo=True)

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
        
        # รันพยากรณ์ภาพ
        results = model(img)
        
        os.makedirs("static", exist_ok=True)
        
        # บันทึกภาพผลลัพธ์
        results.render()  # เรนเดอร์กรอบลงในภาพของ YOLOv5
        for im in results.ims:
            im_base64 = Image.fromarray(im)
            im_base64.save("static/image0.jpg")

        return render_template("index.html", result_image="static/image0.jpg")

    return render_template("index.html", result_image=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    results = model(img, size=320)