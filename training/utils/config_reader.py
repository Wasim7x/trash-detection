import yaml
import json
class ConfigReader:
    
    def __read_yaml(self, file_path):
        """Reads a YAML file and returns its content."""
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
        
    def __read_json(self, file_path):
        """Reads a JSON file and returns its content."""
        with open(file_path, 'r') as file:
            return json.load(file)

    @staticmethod
    def get_config(config_file):
        """Reads a configuration file and returns its content."""
        if config_file.endswith('.yaml') or config_file.endswith('.yml'):
            return ConfigReader().__read_yaml(config_file)
        elif config_file.endswith('.json'):
            return ConfigReader().__read_json(config_file)
        else:
            raise ValueError("Unsupported configuration file format. Use .yaml, .yml, or .json")                
    