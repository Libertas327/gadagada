from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import uuid
from datetime import datetime
import sqlite3
from ultralytics import YOLO

# 初始化Flask应用
app = Flask(__name__)

# 配置：上传文件保存的文件夹
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 允许的文件格式
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}

# 自动创建上传文件夹
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ==================== 加载YOLO模型 ====================
# 模型路径和类别映射，和训练时严格对应
MODEL_PATH = "./weights/best.pt"
CLASS_NAMES = ["gloves", "goggles", "gown"]  # 索引0:手套  索引1:护目镜  索引2:防护服
CONF_THRESHOLD = 0.5  # 置信度阈值，超过判定为佩戴/穿戴

# 全局加载模型，只加载一次，提升速度
model = YOLO(MODEL_PATH)
print("✅ YOLO模型加载完成")

# 工具函数：检查文件格式
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 工具函数：连接数据库
def get_db():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

# 全局异常捕获
@app.errorhandler(Exception)
def handle_exception(e):
    print(f"服务异常：{str(e)}")
    return jsonify({"code":500, "msg":f"服务器内部错误：{str(e)}"}),500

# ==================== 核心接口 ====================
# 首页：返回前端网页
@app.route('/')
def index():
    return render_template('index.html')

# 文件上传 + YOLO识别接口
@app.route('/upload', methods=['POST'])
def upload_file():
    # 1. 检查有没有上传文件
    if 'file' not in request.files:
        return jsonify({'code': 400, 'msg': '没有上传文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'code': 400, 'msg': '文件名为空'}), 400

    # 2. 检查文件格式
    if not allowed_file(file.filename):
        return jsonify({'code': 400, 'msg': '仅支持 png/jpg/jpeg/mp4 格式'}), 400

    # 3. 保存文件，用随机命名避免重名
    ext = file.filename.rsplit('.', 1)[1].lower()
    new_filename = str(uuid.uuid4()) + '.' + ext
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
    file.save(file_path)

    # 4. YOLO真实识别
    results = model(file_path, conf=CONF_THRESHOLD, iou=0.45, imgsz=640)
    boxes = results[0].boxes

    # 初始化每个类别的最高置信度
    class_max_conf = {
        "gloves": 0.0,
        "goggles": 0.0,
        "gown": 0.0
    }
    detections = []

    # 遍历所有检测框（循环内处理每个框）
    for box in boxes:
        cls_id = int(box.cls)
        cls_name = CLASS_NAMES[cls_id]
        conf = float(box.conf)

        # 更新该类别的最高置信度
        if conf > class_max_conf[cls_name]:
            class_max_conf[cls_name] = conf

        # 提取框坐标
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "class_name": cls_name,
            "confidence": round(conf, 4),
            "bbox": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2)
            }
        })

    # 5. 转换为业务结果格式（补上你漏掉的 detect_result）
    detect_result = {
        "goggles": "佩戴" if class_max_conf["goggles"] >= CONF_THRESHOLD else "未佩戴",
        "goggles_conf": round(class_max_conf["goggles"], 4),
        "gloves": "佩戴" if class_max_conf["gloves"] >= CONF_THRESHOLD else "未佩戴",
        "gloves_conf": round(class_max_conf["gloves"], 4),
        "coat": "穿戴" if class_max_conf["gown"] >= CONF_THRESHOLD else "未穿戴",
        "coat_conf": round(class_max_conf["gown"], 4)
    }

    # 6. 判断风险等级
    risk_level = "存在风险" if (detect_result["goggles"] == "未佩戴" 
                                or detect_result["gloves"] == "未佩戴" 
                                or detect_result["coat"] == "未穿戴") else "合规"

    # 7. 保存记录到数据库
    conn = get_db()
    cursor = conn.cursor()
    detect_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO detection_record 
        (detect_time, file_name, goggles, gloves, coat, goggles_conf, gloves_conf, coat_conf, risk_level, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (detect_time, file.filename, 
          detect_result["goggles"], detect_result["gloves"], detect_result["coat"],
          detect_result["goggles_conf"], detect_result["gloves_conf"], detect_result["coat_conf"],
          risk_level, file_path))
    conn.commit()
    conn.close()

    # 8. 返回识别结果给前端
    return jsonify({
        'code': 200,
        'msg': '识别成功',
        'data': {
            'filename': new_filename,
            'detect_time': detect_time,
            'result': detect_result,
            'risk_level': risk_level,
            'detections': detections
        }
    })


# 图片访问接口：预览上传图片
@app.route('/get_img/<filename>')
def get_img(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# 删除单条检测记录接口
@app.route('/del_record/<int:record_id>', methods=['DELETE'])
def del_record(record_id):
    conn = get_db()
    cursor = conn.cursor()
    # 删除数据库记录
    cursor.execute("DELETE FROM detection_record WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()
    return jsonify({"code":200, "msg":"删除成功"})

# 历史记录网页页面
@app.route('/historypage')
def history_page():
    return render_template('history.html')

# 查询历史记录接口
@app.route('/history')
def get_history():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM detection_record ORDER BY detect_time DESC")
    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify({'code': 200, 'data': records})

# ==================== 启动程序 ====================
if __name__ == '__main__':
    print("🚀 后端服务启动中...")
    print("📌 访问地址：http://127.0.0.1:5000")
    app.run(debug=False, port=5000)
