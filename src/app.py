import cv2
import time
import threading
import os
import sqlite3
import requests
from flask import Flask, Response, request, jsonify, render_template
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ========== 【这里填模型同学的推理服务地址，他启动后发给你】 ==========
MODEL_API_URL = "http://127.0.0.1:8000/predict"

# ========== 视频配置、全局变量、线程锁 ==========
CAP_SOURCE = 0
DETECT_INTERVAL = 1.0
frame_lock = threading.Lock()
latest_frame = None
last_detect_time = 0
cap = cv2.VideoCapture(CAP_SOURCE)

# ========== 数据库初始化 ==========
def init_db():
    conn = sqlite3.connect("record.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  img_path TEXT,
                  result TEXT,
                  create_time TEXT)''')
    conn.commit()
    conn.close()

init_db()

# ========== 抓拍保存+入库函数 ==========
def save_and_record(origin_frame, result_text):
    timestr = time.strftime("%Y%m%d_%H%M%S")
    save_name = f"violate_{timestr}.jpg"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
    cv2.imwrite(save_path, origin_frame)

    conn = sqlite3.connect("record.db")
    c = conn.cursor()
    c.execute("INSERT INTO history(img_path, result, create_time) VALUES (?, ?, ?)",
              (save_name, result_text, timestr))
    conn.commit()
    conn.close()
    return save_name

# ========== 调用同学的模型接口，不再本地加载YOLO！ ==========
def detect_frame(cv_frame):
    # 把opencv帧转成jpg二进制，传给模型接口
    ret, buf = cv2.imencode(".jpg", cv_frame)
    img_bytes = buf.tobytes()
    try:
        resp = requests.post(MODEL_API_URL, files={"image": img_bytes}, timeout=5)
        data = resp.json()
        annotated_b64 = data["image"]
        import base64
        img_bin = base64.b64decode(annotated_b64)
        nparr = np.frombuffer(img_bin, np.uint8)
        annotated_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        info_str = data["result"]
        # 违规判断标签，和模型那边标签保持一致
        is_violate = "no_helmet" in info_str
        return annotated_frame, info_str, is_violate
    except Exception as e:
        print("模型接口调用失败：", e)
        return cv_frame, "模型服务未连接", False

# ========== 后台视频线程 ==========
def video_loop():
    global latest_frame, last_detect_time
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.2)
            continue
        now = time.time()
        if now - last_detect_time > DETECT_INTERVAL:
            annotated, info, violate = detect_frame(frame)
            last_detect_time = now
            if violate:
                save_and_record(annotated, info)
            out_frame = annotated
        else:
            out_frame = frame

        with frame_lock:
            latest_frame = out_frame.copy()

t = threading.Thread(target=video_loop, daemon=True)
t.start()

# ========== 视频流生成 ==========
def gen_stream():
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.05)
                continue
            ret, buffer = cv2.imencode(".jpg", latest_frame)
            if not ret:
                continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ===================== 全部路由区 =====================
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/history')
def history_page():
    return render_template("history.html")

@app.route('/monitor')
def monitor():
    return render_template("monitor.html")

@app.route("/upload", methods=["POST"])
def upload_img():
    file = request.files.get("file")
    if not file:
        return jsonify({"ok":False,"msg":"没有文件"})
    timestr = time.strftime("%Y%m%d_%H%M%S")
    fname = f"upload_{timestr}.jpg"
    savep = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    file.save(savep)
    # 读取上传图片，同样调用模型接口
    frame = cv2.imread(savep)
    annotated, info_str, is_violate = detect_frame(frame)
    outname = f"res_{timestr}.jpg"
    outpath = os.path.join(app.config['UPLOAD_FOLDER'], outname)
    cv2.imwrite(outpath, annotated)
    if is_violate:
        save_and_record(annotated, info_str)
    return jsonify({"ok":True,"msg":info_str,"imgname":outname})

@app.route("/api/get_history")
def get_history():
    conn = sqlite3.connect("record.db")
    rows = conn.execute("SELECT * FROM history ORDER BY id DESC").fetchall()
    conn.close()
    data = []
    for r in rows:
        data.append({
            "id": r[0],
            "img_path": r[1],
            "result": r[2],
            "time": r[3]
        })
    return jsonify(data)

@app.route("/api/del_record/<int:rid>", methods=["POST"])
def del_record(rid):
    conn = sqlite3.connect("record.db")
    conn.execute("DELETE FROM history WHERE id=?", (rid,))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})

@app.route("/uploads/<filename>")
def preview_img(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/video_feed")
def video_feed():
    return Response(gen_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")
# ======================================================

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        cap.release()
