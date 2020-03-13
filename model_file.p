import os

def pickle_file(file_loc):
    model_file = [file_path for file_path in os.listdir(os.getcwd()) if file_path.endswith(".pkl")][0]
    model_filePath = os.path.join(file_loc,model_file)
    return(model_filePath)
        
        
