import os
def pickle_file(current_dir_path):
    for file_path in os.listdir(current_dir_path):
        if file_path.endswith(".pkl"):
            return file_path