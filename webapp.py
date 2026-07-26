import os
from ultralytics import YOLO
from PIL import Image
from flask import Flask, render_template, request, redirect

# โหลดโมเดลด้วย ultralytics โดยตรง (ไม่พึ่ง torch.hub แล้ว)
model = YOLO('best.pt')

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
        results = model(img, conf=0.5, iou=0.4)
        
        os.makedirs("static", exist_ok=True)
        
        for r in results:
            im_array = r.plot()
            im = Image.fromarray(im_array[..., ::-1])
            im.save("static/image0.jpg")

        return render_template("index.html", result_image="static/image0.jpg")

    return render_template("index.html", result_image=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)