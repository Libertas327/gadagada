import sqlite3
# 连接或者新建数据库
conn = sqlite3.connect('database.db')
cursor = conn.cursor()
# 创建检测记录表
cursor.execute('''
CREATE TABLE IF NOT EXISTS detection_record (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detect_time TEXT,
    file_name TEXT,
    goggles TEXT,
    gloves TEXT,
    coat TEXT,
    goggles_conf REAL,
    gloves_conf REAL,
    coat_conf REAL,
    risk_level TEXT,
    image_path TEXT
)
''')
conn.commit()
conn.close()
print("数据库和数据表创建完成！")
