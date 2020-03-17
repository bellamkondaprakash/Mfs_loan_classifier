

def get_static_data(dir_path):
    import os
    pkl_file = [ file_path for file_path in os.listdir(os.getcwd()) if file_path.endswith('.sav')]
    model_file_path = os.path.join(dir_path,pkl_file)
    return(model_file_path)