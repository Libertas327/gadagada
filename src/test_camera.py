import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 摄像头打开失败，请检查：相机权限、摄像头遮挡、被其他软件占用，也可以尝试把 0 改成 1 或 2")
else:
    print("✅ 摄像头打开成功！")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Camera Test（按 q 关闭窗口）", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
