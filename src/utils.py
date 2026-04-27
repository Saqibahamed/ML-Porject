import sys
import os
import dill

from src.exception import CustomException


def save_file(file_path,obj):
    try:
        file_dir = os.path.dirname(file_path)

        os.makedirs(file_dir,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)