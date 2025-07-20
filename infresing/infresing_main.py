import os
import sys
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

sys.path[0] = str(Path(__file__).resolve().parent.parent)
# from custom_model import CustomResNet18
from training.utils.config_reader import ConfigReader
from training.utils.custom_transform import get_test_val_transforms
from training.model_initialization.resnet50 import ResNet50Transfer


transform = get_test_val_transforms()

MODEL_PATH = 'model\\ResNet50.pt'


class TrashDetectionModel:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = ConfigReader.get_config(config_path)
        self.data_path = self.config.get("DATA_DIR", "data")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data directory not found at {self.data_path}")
        self.batch_size = self.config.get("batch_size", 32)
        self.num_classes = self.config.get("num_classes", 10)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = self.config.get("MODEL_PATH", "model/model.pt")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        self.class_names = self.config.get("CLASS_NAMES")  
    
   
    def pridiction(self):

        # ---------- Load model ----------
        model = ResNet50Transfer(num_classes=10).to(self.device)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.eval()
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_count = 0

        for root, _, files in os.walk(self.data_path):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_path = os.path.join(root, file)

                    try:
                        # Load and preprocess
                        image = Image.open(image_path).convert('RGB')
                        input_tensor = transform(image).unsqueeze(0).to(self.device)

                        # Predict
                        with torch.no_grad():
                            output = model(input_tensor)
                            _, predicted = torch.max(output, 1)
                            pred_class = self.class_names[predicted.item()]

                        # Draw prediction on image
                        draw = ImageDraw.Draw(image)
                        font = ImageFont.truetype("arial.ttf", 24) if os.name == 'nt' else None
                        draw.text((10, 10), f"Predicted: {pred_class}", fill='red', font=font)

                        # # Save result
                        # save_path = os.path.join(OUTPUT_DIR, f"{image_count}_{file}")
                        # image.save(save_path)

                        # Show result
                        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        cv2.imshow("Prediction", cv_image)
                        cv2.waitKey(500)

                        image_count += 1

                    except Exception as e:
                        print(f"[ERROR] Failed to process {image_path}: {e}")

        cv2.destroyAllWindows()
        print(f"✅ Done. Processed {image_count} images.")


if __name__ == "__main__":
    config_path = r"infresing\\config\\config.yml"
    model = TrashDetectionModel(config_path)
    model.pridiction()        