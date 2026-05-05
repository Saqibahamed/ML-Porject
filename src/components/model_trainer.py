from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

import logging
from src import logger
import sys
import os
from dataclasses import dataclass
from src.utils import evaluate_model, save_file
from src.exception import CustomException

@dataclass
class ModelTrainerConfig:
    trained_model_data_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
       self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):

        try:
            logging.info("split train test input data")

            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                        'Linear Regression': LinearRegression(),
                        'Ridge': Ridge(),
                        'Lasso' : Lasso(),
                        'SVR' : SVR(),
                        'KNN' : KNeighborsRegressor(),
                        'Decision Tree' : DecisionTreeRegressor(),
                        'Random Forest' : RandomForestRegressor(),
                        'Ada Boost' : AdaBoostRegressor(),
                        'Gradient Boost' : GradientBoostingRegressor(),
                        'Cat Boost' : CatBoostRegressor(),
                        'XGB' : XGBRegressor()
                    }
                 
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            ## to get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## to get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best mode found")
            
            logging.info("Best found model on both training and testing dataset")

            save_file(
                file_path=self.model_trainer_config.trained_model_data_path,
                obj= best_model
                )

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test,predicted)

            return r2 

            
        except Exception as e:
            raise CustomException(e,sys)
                  


