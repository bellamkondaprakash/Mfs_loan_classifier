

def api_test(args):
    import os
    pkl_file = [ os.path.join(file_path) for file_path in os.listdir(os.getcwd()) if file_path.endswith('.sav')][0]
    model_file_path = os.path.join(args,os.path.basename(pkl_file))
    
    return(model_file_path)