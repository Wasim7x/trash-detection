import os
import sys
from pathlib import Path
import yaml
import torch
from torchsummary import summary
# from tensorflow.keras.callbacks import EarlyStopping

sys.path[0] = str(Path(__file__).resolve().parent.parent)

from training.utils.config_reader import ConfigReader
from training.utils.common_fun import CommonFun
from training.data_ingestion.ingest_data import DataIngestor
from training.data_ingestion.data_download import ImageDownloader
from training.model_initialization.resnet50 import ResNet50Transfer
from training.modle_training.training import ModelTraining

DATA_PATH = Path(Path(__file__).resolve().parent.parent.parent,"garbage-dataset")
fig_path = os.path.join(str(Path(__file__).parent.parent),"figures","Training_matrics.jpg")
MODEL_PATH = os.path.join(str(Path(__file__).parent.parent),"model","Resnet50.pt")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(fig_path), exist_ok=True)

class TrainingOrchestrator:
    def __init__(self, config_file: str | Path):
        self.config_file = config_file
        self.config = ConfigReader.get_config(config_file)
        self.dataset_classes = DataIngestor(self.config_file).get_class_names()
        self.num_epochs = self.config.get("num_epochs")
        self.lr= self.config.get("learnig_rate")
        self.patience = self.config.get("patience")

    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")        

    def run_training_pipeline(self):
        """
        Main method to run the training pipeline.
        This method orchestrates the entire training process based on the configuration.
        """
        ### load dataset ###
        if not os.path.exists(DATA_PATH):
            ImageDownloader(self.config_file).lodad_images_local()
        loader = DataIngestor(self.config_file)
        num_classes = loader.get_num_classes()
        print("num of classes is :",num_classes)
        train_loader, test_loader, val_loader = loader.get_dataloaders()

        # Step 3: Train Model
        device = self.get_device()
        print(f'Model is traing using {device}')
        model = ResNet50Transfer(num_classes).to(device)
        summary(model, (3,224,224))
        
        model_training_obj = ModelTraining(self.get_device())
        model_weights, training_results = model_training_obj.train_model(model, train_loader, test_loader, self.num_epochs, self.lr,self.patience)

        # Step 4: Save Model and training results
        
        torch.save(model_weights, MODEL_PATH)
        print(f"Model saved at {MODEL_PATH}")

        print("training result type:",type(training_results.values()))
        print("training result:",list(training_results.values())[0])

        x_data = range(1, len(list(training_results.values())[0])+1)
        label_ = ('Epochs', 'Loss or Accuracy')
        CommonFun.plot_graph(x_data, training_results, label_, 'Training_matrics.jpg')


        # # Step 5: Evaluate model on val set
        self.model_evaluation(val_loader)

    def model_evaluation(self, val_data):
        model = ResNet50Transfer(num_classes=10).to(self.get_device())
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        val_acc, val_loss = ModelTraining(self.get_device()).evaluate_model(model, val_data, self.dataset_classes, image_save_dir = 'training/validation_output')
        print(f"Val Acc: {val_acc:.2f}, Val Loss: {val_loss:.4f}%")
        print("-" * 40)
           

def main():
    """
    Main entry point for the training script.
    Reads the configuration file and starts the training orchestrator.
    """
    config_file = r'training\\configs\\config.yaml'  # Replace with your actual config file path
    if not os.path.exists(config_file):
        print(f"Configuration file {config_file} does not exist.")
        sys.exit(1)

    orchestrator = TrainingOrchestrator(config_file)
    orchestrator.run_training_pipeline()


if __name__ == "__main__":
    main()                
        
