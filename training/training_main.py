import os
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parent.parent)
from training.utils.config_reader import ConfigReader

class TrainingOrchestrator:
    def __init__(self, config_file):
        self.config = ConfigReader.get_config(config_file)

    def run_training_pipeline(self):
        """
        Main method to run the training pipeline.
        This method orchestrates the entire training process based on the configuration.
        """
        # Here you would implement the logic to run the training pipeline
        # For example, initializing datasets, models, and starting the training loop
        print("Running training pipeline with the following configuration:")
        print(self.config)

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
        
