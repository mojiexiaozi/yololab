from yololab import YOLO

if __name__ == "__main__":
    for m in [
        # "light-cls-tf_efficientnet_b0",
        # "light-cls-mobilenetv3_large",
        # "yolov8n-cls",
        "yolov8s-cls",
        "yolov8m-cls",
        "yolov8l-cls",
    ]:
        model = YOLO(f"{m}.yaml", task="classify", verbose=True)  # 从头开始构建新模型

        model.train(
            data="datasets/imagenet10",
            epochs=500,
            nc=15,
            device=0,
            imgsz=224,
            batch=16,
            project=f"runs/classify/{m}",
        )

    # model = YOLO("yolov8n-cls.pt", verbose=True)  # 加载预训练模型（建议用于训练
    # metrics = model.val()  # 在验证集上评估模型性能
    # results = model("https://ultralytics.com/images/bus.jpg")  # 对图像进行预测
    # success = model.export(format="onnx")  # 将模型导出为 ONNX 格式
