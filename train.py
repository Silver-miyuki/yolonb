from ultralytics import YOLO

# 加载模型
model = YOLO("yolov8n-pose.pt")

if __name__ == '__main__':
    print("开始GPU训练...")
    model.train(
        # 你的yaml路径
        data=r"C:\Users\Silver\Desktop\hockey-pose-dataset\hockeydataset\data.yaml",
        epochs=10,
        imgsz=640,
        batch=4,          # GPU显存够的话，直接调大，不用再用1了
        device=0,         # 用GPU，0代表电脑上的第一个显卡
        augment=False,
        workers=0
    )
    print("----- 训练结束 -----")