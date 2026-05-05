import sys
import os
import dill

from src.exception import CustomException
from sklearn.metrics import r2_score


def save_file(file_path,obj):
    try:
        file_dir = os.path.dirname(file_path)

        os.makedirs(file_dir,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report={}

        for name,model in models.items():
            
            model.fit(X_train,y_train) # train model

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[name] = test_model_score

        return report



    except Exception as e:
        raise CustomException(e,sys)



