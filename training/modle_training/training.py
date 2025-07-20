import os
import torch
import copy
import sys
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
import torchvision.transforms.functional as F
from utils.common_fun import CommonFun
from utils.config_reader import ConfigReader

class ModelTraining:
    def __init__(self, device='cpu'):
        self.device = device
        self.criterion = nn.CrossEntropyLoss() 
        self.counter = 0
        self.best_loss = float('inf')
        self.model_weights = None   

    def train_model(self, model, train_loader, val_loader, num_epochs, lr,patience):
        # Use CrossEntropyLoss (handles softmax internally)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model.to(self.device)
        training_results = defaultdict(list)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            # Training loop
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            train_acc = 100 * correct / total
            train_loss = running_loss/len(train_loader)
            val_acc, val_loss = self.evaluate_model(model, val_loader, self.criterion)

            training_results['train_loss'].append(train_loss)
            training_results['val_loss'].append(val_loss)
            training_results['train_accuracy'].append(train_acc)
            training_results['val_accuracy'].append(val_acc)

            print(f"Epoch {epoch+1} Summary:")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print("-" * 40)

            #early stoping
            if val_loss < self.best_loss - 5:
                self.best_loss = val_loss
                self.counter = 0
                self.model_weights = copy.deepcopy(model.state_dict())
            else:
                self.counter += 1
                if self.counter >= patience:
                    print("Early stopping triggered.")
                    break

        return self.model_weights, training_results
    

    def evaluate_model(self, model, dataloader, dataset_classes:list, criterion=nn.CrossEntropyLoss(), image_save_dir=None):
        common_func_obj = CommonFun()
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

                if image_save_dir:
                    for i in range(images.size(0)):
                        img_tensor = images[i].cpu()
                        pred_label = predicted[i].item()

                        # set output image save path 
                        img_save_path = os.path.join(image_save_dir, dataset_classes[pred_label])
                        common_func_obj.create_dir(img_save_path)
                        image_number = len([f for f in Path(img_save_path).iterdir() if f.is_file()])
                        image_file = os.path.join(img_save_path, f"img_{image_number}.jpg")

                        # Un-normalize and convert to PIL image
                        img = F.to_pil_image(common_func_obj.unnormalize_tensor(img_tensor))
                        img = common_func_obj.draw_label(img, dataset_classes[pred_label])
                        common_func_obj.save_images(img, image_file)

        acc = 100 * correct / total
        avg_loss = total_loss / len(dataloader)
        return acc, avg_loss