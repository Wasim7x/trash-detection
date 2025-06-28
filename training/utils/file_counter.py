import os

class FileCounter:
    """
    A utility class to count files in a directory.
    """
    
    @staticmethod
    def count_files(directory: str) -> int:
        """
        Counts the number of files in the specified directory.
        
        """
        total_files = 0
        for root, dirs, files in os.walk(directory):
            total_files += len(files)
        return total_files