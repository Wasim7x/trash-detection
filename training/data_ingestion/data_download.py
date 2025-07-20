import os
import sys
from pathlib import Path
import kagglehub
import shutil
import yaml


sys.path[0] = str(Path(__file__).resolve().parent.parent)

from utils.config_reader import ConfigReader
from utils.file_counter import FileCounter

destination_path = Path(Path(__file__).resolve().parent.parent.parent) 

class ImageDownloader:
    """
    downloads images from a specified source and saves them to working directory
    """
    def __init__(self, config_file):
        self.config = ConfigReader.get_config(config_file)
        self.verbose = self.config.get("verbose", False)
        self.path = self.download_images()
        self.destination_path = destination_path
    
    def download_images(self) -> str:
        """
        Downloads images based on the configuration.
        This method should implement the logic to download images from the specified source.
        """
        print("Downloading images...")
        dataset_id = str(self.config.get("dataset_id"))
        
        if not dataset_id:
            print("No dataset_id found in the configuration.")
            return
        
        path = str(kagglehub.dataset_download(dataset_id))

        if not path:
            raise FileExistsError("Dataset download failed. Please check the dataset_id and your internet connection.")

        if self.verbose:
            print(f"dataset downloaded from: {dataset_id}")
            print(f"Dataset downloaded to: {path}")
        print(path)
        return path

 

    def lodad_images_local(self):
        source_path = Path(self.path)
        
        print("destination" ,self.destination_path)
        try:
            shutil.copytree(src=source_path, dst=self.destination_path, dirs_exist_ok=True)
            # with open(self.config_file, 'w') as f:
            #     yaml.dump({"output_path": str(destination_path)}, f)
        except Exception as e:
            print(f"❌ Failed to copy folder: {e}")

        if self.verbose:
            print(f"Images copied to: {destination_path}")
        

def main():
    config_file = r"training\\configs\\config.yaml" 
    if not os.path.exists(config_file):
        print(f"Configuration file {config_file} does not exist.")
        sys.exit(1)
    destination_path = Path(Path(__file__).resolve().parent.parent.parent)        
    ImageDownloader(config_file).lodad_images_local(destination_path)
   
    # no_of_images = FileCounter.count_files(os.path.join(str(destination_path), 'garbage-dataset'))
    
    print("{no_of_images} Images download completed successfully.") 

if __name__ == "__main__":
    main()

   