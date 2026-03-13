from ultralytics import YOLO
import torch

def main():

    # Check if GPU is available
    if torch.cuda.is_available():
        device = 0
        print("✅ GPU detected:", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        print("⚠ GPU not detected, using CPU")

    # Load pretrained YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Train the model
    model.train(
        data="configs/dataset.yaml",   # dataset config
        epochs=40,                     # training epochs
        imgsz=640,                     # image size
        batch=16,                      # batch size (good for RTX 3050)
        device=device,                 # GPU
        project="outputs",             # output folder
        name="smartwastenet_training", # experiment name
        workers=4,                     # faster data loading
        optimizer="auto",              # optimizer selection
        patience=10                    # early stopping
    )

    print("\n🎉 Training finished!")

if __name__ == "__main__":
    main()