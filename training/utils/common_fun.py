import os
import sys
from PIL import ImageDraw, ImageFont
from pathlib import Path
import matplotlib.pyplot as plt
import random

sys.path[0] = str(Path(__file__).resolve().parent.parent)

def __init__(self):
    pass
    
class CommonFun:
    @staticmethod
    def create_dir(path):
       """Create a directory if it does not exist."""
       if not os.path.exists(path):
           os.makedirs(path)

    @staticmethod
    def get_absolute_path(relative_path: str) -> str: 
        """
        Get the absolute path of a given relative path
        """      
        return str(Path(__file__).resolve().parent.parent / relative_path)  
    
    @staticmethod
    def write_to_yaml(file_path: str, data: dict):
        """
        Write a dictionary to a YAML file.
        """
        import yaml
        with open(file_path, 'w') as file:
            yaml.safe_dump(data, file, default_flow_style=False)

    @staticmethod
    def unnormalize_tensor(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor.clamp(0, 1)


    @staticmethod
    def draw_label(image, label, font=ImageFont.load_default()):
        """It draw the level on the image."""
        draw = ImageDraw.Draw(image)
        draw.text((image.width - 80, 5), str(label), font=font, fill=(0, 0, 255))
        return image
    
    @staticmethod
    def save_images(image, file_path):
        image.save(file_path)
    
    @staticmethod
    def plot_graph(x_data:list, y_data:dict|list, label_:tuple, title_:str=None, save_path:str=None):
        """
        This function is used to draw graph and save it. 
        """

        plt.figure(figsize=(8, 5))
        random.seed(42)
        training_metrics = str
        if isinstance(y_data, dict):
            for key, val in y_data.items():
                training_metrics = str(training_metrics) + ' vs ' + str(key)
                color_ = (random.random(), random.random(), random.random())
                plt.plot(x_data, val, color=color_, label=key, linewidth=2)

        if title_ is not None: plt.title(title_) 
        elif isinstance(y_data, dict): plt.title(f'training Metrics: {training_metrics}')
        else: plt.title(f'{label_[0]} vs {label_[1]}')

        plt.xlabel(label_[0])
        plt.ylabel(label_[1])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi = 300) 

        plt.show()
       


           