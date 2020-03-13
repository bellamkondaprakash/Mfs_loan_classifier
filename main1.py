import os

def get_model(dir_path):
    pkl_file = [ file_path for file_path in os.listdir(os.getcwd()) if file_path.endswith('.pkl')][0]
    model_file_path = os.path.join(dir_path,pkl_file)
    return(model_file_path)

