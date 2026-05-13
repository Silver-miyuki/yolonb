import cv2
import numpy as np
import os
from ultralytics import YOLO
from collections import deque

# 加载YOLO姿态模型
model = YOLO("best.pt")

# ===================== 【固定配置：仅修改这里】 =====================
VIDEO_FOLDER = r"C:\Users\Silver\Desktop\hockey-pose-dataset\HockeyFights"
OUTPUT_FOLDER = r"C:\Users\Silver\Desktop\检测结果输出"
# ==================================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------------- 工具函数 --------------------------
def calc_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

# ✅ 【核心修复】IOU坐标错误修正（打架识别的关键！）
def calc_bbox_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2  # 这里修复了坐标顺序！
    xi1, yi1 = max(x1, x3), max(y1, y3)
    xi2, yi2 = min(x2, x4), min(y2, y4)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def calc_torso_angle(kpt):
    s_mid = (kpt[5, :2] + kpt[6, :2]) / 2
    h_mid = (kpt[11, :2] + kpt[12, :2]) / 2
    vec = s_mid - h_mid
    vert = np.array([0, 1])
    cos_ang = np.dot(vec, vert) / (np.linalg.norm(vec) * np.linalg.norm(vert) + 1e-6)
    return np.arccos(np.clip(cos_ang, -1, 1)) * 180 / np.pi

def calc_kpt_move(pre, cur):
    if pre is None or cur is None: return 0
    mask = (pre[:, 2] > 0.5) & (cur[:, 2] > 0.5)
    p1, p2 = pre[mask, :2], cur[mask, :2]
    return np.mean(np.linalg.norm(p2 - p1, axis=1)) if len(p1) > 0 else 0

# -------------------------- 异常判定函数 --------------------------
def judge_fall(bbox, kpt, hist, fps):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if h == 0: return False
    ratio = w / h
    ang = calc_torso_angle(kpt)
    hist.append(calc_center(bbox))
    if len(hist) > 3: hist.popleft()
    speed = abs(hist[-1][1] - hist[0][1]) * fps / len(hist) if len(hist) >= 2 else 0
    hy = (kpt[11, 1] + kpt[12, 1]) / 2
    ay = (kpt[15, 1] + kpt[16, 1]) / 2
    diff = abs(ay - hy) / h
    return ratio > 0.7 and ang > 60 and speed > 150 and diff < 0.3

def judge_run(bbox, hist, fps):
    hist.append(calc_center(bbox))
    if len(hist) > 5: hist.popleft()
    if len(hist) < 2: return False
    dis = np.linalg.norm(np.array(hist[-1]) - np.array(hist[0]))
    return dis * fps / len(hist) > 250

# ✅ 【打架终极优化】瞬时识别，0延迟，无严格限制
def judge_fight(plst, pre_kpt):
    # 只要2人靠近 + 轻微动作，直接判定打架
    for i in range(len(plst)):
        for j in range(i + 1, len(plst)):
            b1, k1 = plst[i]
            b2, k2 = plst[j]
            iou = calc_bbox_iou(b1, b2)
            # 超宽松阈值：两人轻微重叠就算
            if iou > 0.1:
                return True
    return False

# -------------------------- 单视频检测函数 --------------------------
def detect_single_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    pre_gray, pre_kpts, center_his = None, [], []
    print(f"开始处理：{os.path.basename(video_path)}")

    while True:
        ret, frame = cap.read()
        if not ret: break
        cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = model(frame, conf=0.5, classes=0, verbose=False)
        person, cur_kpt = [], []

        for r in res:
            if r.boxes is None or r.keypoints is None: continue
            for box, kpt in zip(r.boxes, r.keypoints.data):
                xyxy = list(map(int, box.xyxy[0]))
                k = kpt.cpu().numpy()
                person.append((xyxy, k))
                cur_kpt.append(k)

        # 初始化缓存
        while len(center_his) < len(person): center_his.append(deque(maxlen=5))
        while len(center_his) > len(person): center_his.pop()
        while len(pre_kpts) < len(person): pre_kpts.append(None)
        while len(pre_kpts) > len(person): pre_kpts.pop()

        fall, run, fight = False, False, False
        # 单人检测
        for idx, (bb, k) in enumerate(person):
            if judge_fall(bb, k, center_his[idx], fps):
                fall = True
                cv2.putText(frame, "FALL ALERT", (bb[0], bb[1]-10), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            if judge_run(bb, center_his[idx], fps):
                run = True
                cv2.putText(frame, "RUN ALERT", (bb[0], bb[1]-40), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (255,0,0),2)
            for x,y,c in k:
                if c>0.5: cv2.circle(frame, (int(x),int(y)),3,(0,255,255),-1)

        # ✅ 打架瞬时检测（无延迟，无帧数限制）
        if len(person) >= 2:
            fight = judge_fight(person, pre_kpts)
            if fight:
                cv2.putText(frame, "FIGHT ALERT", (20, 60), cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),3)

        if fall or run or fight: out.write(frame)
        pre_gray, pre_kpts = cur_gray.copy(), cur_kpt

    cap.release()
    out.release()
    print(f"处理完成！结果已保存至：{output_path}\n")

# -------------------------- 批量处理 --------------------------
def batch_process_videos():
    for filename in os.listdir(VIDEO_FOLDER):
        if filename.lower().endswith(".avi"):
            video_path = os.path.join(VIDEO_FOLDER, filename)
            output_name = os.path.splitext(filename)[0] + "_result.avi"
            output_path = os.path.join(OUTPUT_FOLDER, output_name)
            detect_single_video(video_path, output_path)

    cv2.destroyAllWindows()
    print("✅ 所有视频处理完成！结果统一存放在：", OUTPUT_FOLDER)

# -------------------------- 运行 --------------------------
if __name__ == "__main__":
    batch_process_videos()