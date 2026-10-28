from flask import Flask, request, jsonify, send_from_directory, render_template, Response, redirect, url_for
import os
import uuid
from datetime import datetime
import sqlite3
import time
import cv2
from ultralytics import YOLO

# 初始化Flask应用
app = Flask(__name__)
# 配置：上传文件保存的文件夹
UPLOAD_FOLDER = 'uploads'
ALARM_SNAP_FOLDER = "alarm_snap"  # 告警截图保存目录
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 自动创建文件夹
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(ALARM_SNAP_FOLDER):
    os.makedirs(ALARM_SNAP_FOLDER)

# 允许的文件格式
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}

# ==================== 加载YOLO模型 ====================
MODEL_PATH = "./weights/best.pt"
CLASS_NAMES = ["gloves", "goggles", "gown"]
CONF_THRESHOLD = 0.5

# ========== 摄像头实时流配置 ==========
CAP_SOURCE = 0
ALERT_INTERVAL = 30
last_alert_time = 0

model = YOLO(MODEL_PATH)
print("✅ YOLO模型加载完成")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_db():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

# 保存告警记录，默认audit_status='pending'待审批
def save_alarm_snapshot(frame, lack_goggles, lack_gloves, lack_coat):
    snap_filename = f"alarm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    snap_full_path = os.path.join(ALARM_SNAP_FOLDER, snap_filename)
    cv2.imwrite(snap_full_path, frame)

    conn = get_db()
    cur = conn.cursor()
    alarm_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute('''
    INSERT INTO alarm_record (alarm_time, alarm_img_path, lack_goggles, lack_gloves, lack_coat, remark, audit_status)
    VALUES (?,?,?,?,?,?,?)
    ''',(alarm_time, snap_full_path, lack_goggles, lack_gloves, lack_coat, "待安全员核查", "pending"))
    conn.commit()
    conn.close()

@app.errorhandler(Exception)
def handle_exception(e):
    print(f"服务异常：{str(e)}")
    return jsonify({"code":500, "msg":f"服务器内部错误：{str(e)}"}),500

# 首页
@app.route('/')
def index():
    return render_template('index.html')

# 告警审核页面
@app.route('/audit/<int:alarm_id>')
def audit_page(alarm_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM alarm_record WHERE id = ?",(alarm_id,))
    record = cur.fetchone()
    conn.close()
    if not record:
        return "告警记录不存在",404
    return render_template("audit_alarm.html", item=dict(record))

# 设置审核状态接口 pass / reject
@app.route('/set_audit/<int:alarm_id>',methods=['POST'])
def set_audit(alarm_id):
    data = request.get_json()
    status = data.get("status")
    if status not in ["pass","reject"]:
        return jsonify({"code":400,"msg":"状态非法"})
    conn = get_db()
    cur = conn.cursor()
    cur.execute("UPDATE alarm_record SET audit_status=? WHERE id=?",(status,alarm_id))
    conn.commit()
    conn.close()
    return jsonify({"code":200,"msg":"审核完成"})

# 文件上传识别
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'code': 400, 'msg': '没有上传文件'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'code': 400, 'msg': '文件名为空'}), 400
    if not allowed_file(file.filename):
        return jsonify({'code': 400, 'msg': '仅支持 png/jpg/jpeg/mp4 格式'}), 400

    ext = file.filename.rsplit('.', 1)[1].lower()
    new_filename = str(uuid.uuid4()) + '.' + ext
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
    file.save(file_path)

    results = model(file_path, conf=CONF_THRESHOLD, iou=0.45, imgsz=640)
    boxes = results[0].boxes
    class_max_conf = {"gloves":0.0,"goggles":0.0,"gown":0.0}
    detections = []
    for box in boxes:
        cls_id=int(box.cls)
        cls_name=CLASS_NAMES[cls_id]
        conf=float(box.conf)
        if conf>class_max_conf[cls_name]:
            class_max_conf[cls_name]=conf
        x1,y1,x2,y2 = box.xyxy[0].tolist()
        detections.append({
            "class_name":cls_name,
            "confidence":round(conf,4),
            "bbox":{"x1":round(x1,2),"y1":round(y1,2),"x2":round(x2,2),"y2":round(y2,2)}
        })
    detect_result = {
        "goggles":"佩戴" if class_max_conf["goggles"]>=CONF_THRESHOLD else "未佩戴",
        "goggles_conf":round(class_max_conf["goggles"],4),
        "gloves":"佩戴" if class_max_conf["gloves"]>=CONF_THRESHOLD else "未佩戴",
        "gloves_conf":round(class_max_conf["gloves"],4),
        "coat":"穿戴" if class_max_conf["gown"]>=CONF_THRESHOLD else "未穿戴",
        "coat_conf":round(class_max_conf["gown"],4)
    }
    risk_level = "存在风险" if (detect_result["goggles"]=="未佩戴" or detect_result["gloves"]=="未佩戴" or detect_result["coat"]=="未穿戴") else "合规"
    conn = get_db()
    cursor = conn.cursor()
    detect_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
    INSERT INTO detection_record (detect_time, file_name, goggles, gloves, coat, goggles_conf, gloves_conf, coat_conf, risk_level, image_path)
    VALUES (?,?,?,?,?,?,?,?,?,?)
    ''',(detect_time,file.filename,detect_result["goggles"],detect_result["gloves"],detect_result["coat"],
         detect_result["goggles_conf"],detect_result["gloves_conf"],detect_result["coat_conf"],risk_level,file_path))
    conn.commit()
    conn.close()
    return jsonify({
        'code':200,'msg':'识别成功',
        'data':{
            'filename':new_filename,
            'detect_time':detect_time,
            'result':detect_result,
            'risk_level':risk_level,
            'detections':detections
        }
    })

@app.route('/get_img/<filename>')
def get_img(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/get_alarm_img/<filename>')
def get_alarm_img(filename):
    return send_from_directory(ALARM_SNAP_FOLDER, filename)

# 获取【待审批】告警列表，首页用
@app.route('/get_pending_alarm')
def get_pending_alarm():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM alarm_record WHERE audit_status='pending' ORDER BY alarm_time DESC")
    alarm_list = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify({"code":200,"data":alarm_list})

# 获取全部告警
@app.route('/get_alarm_list')
def get_alarm_list():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM alarm_record ORDER BY alarm_time DESC")
    alarm_list = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify({"code":200,"data":alarm_list})

def generate_camera_frames():
    global last_alert_time
    cap = cv2.VideoCapture(CAP_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    if not cap.isOpened():
        print("摄像头打开失败，请检查权限、占用、索引号")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=CONF_THRESHOLD, iou=0.45, imgsz=640)
        boxes = results[0].boxes
        class_max_conf = {"gloves":0.0,"goggles":0.0,"gown":0.0}
        for box in boxes:
            cls_id=int(box.cls)
            cls_name=CLASS_NAMES[cls_id]
            conf=float(box.conf)
            if conf>class_max_conf[cls_name]:
                class_max_conf[cls_name]=conf
            x1,y1,x2,y2 = map(int,box.xyxy[0].tolist())
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        goggles_ok = class_max_conf["goggles"]>=CONF_THRESHOLD
        gloves_ok = class_max_conf["gloves"]>=CONF_THRESHOLD
        coat_ok = class_max_conf["gown"]>=CONF_THRESHOLD
        has_risk = not(goggles_ok and gloves_ok and coat_ok)
        now_ts = time.time()
        if has_risk and (now_ts-last_alert_time>ALERT_INTERVAL):
            last_alert_time = now_ts
            cv2.putText(frame,"⚠安全风险，请安全员注意！",(10,35),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
            lack_goggles = "未佩戴" if not goggles_ok else "已佩戴"
            lack_gloves = "未佩戴" if not gloves_ok else "已佩戴"
            lack_coat = "未穿戴" if not coat_ok else "已穿戴"
            save_alarm_snapshot(frame, lack_goggles, lack_gloves, lack_coat)
        _, jpeg_buffer = cv2.imencode('.jpg',frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg_buffer.tobytes() + b'\r\n')
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_camera_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_alert')
def get_alert():
    global last_alert_time
    return jsonify({"code":200,"last_alert":last_alert_time,"alert_interval":ALERT_INTERVAL})

if __name__ == '__main__':
    print("🚀 后端服务启动中...")
    print("📌 访问地址：http://127.0.0.1:5000")
    app.run(debug=False, port=5000)
