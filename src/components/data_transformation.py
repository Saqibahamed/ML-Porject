import sys
import os
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from src.utils import save_file

@dataclass
class DataTransformationConfig:
    preprocessing_obj_data_path=os.path.join('artifacts','preprocessing.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            logging.info("Initializing Feature encoding and transformation")
            num_columns=['reading score','writing score']
            cat_columns=['gender','race/ethnicity','parental level of education','lunch','test preparation course']

            num_pipeline = Pipeline(
                [
                    ('impute',SimpleImputer(strategy='median')),
                    ('StandardScaler',StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                [
                    ('impute',SimpleImputer(strategy='most_frequent')),
                    ('OneHotEncoding',OneHotEncoder(drop='first'))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,num_columns),
                    ('cat_pipeline',cat_pipeline,cat_columns)
                ]
            )

            logging.info("feature transformation completed")

            return preprocessor
        




        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessing_obj = self.get_data_transformer_obj()

            target_column = 'math score'

            input_feature_train_df = train_df.drop(target_column,axis=1) 
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(target_column,axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("applying preprocessing on train and test data")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr =  np.c_[
                (input_feature_train_arr),
                (target_feature_train_df)
            ]
           

            test_arr = np.c_[
                (input_feature_test_arr),
                (target_feature_test_df)
            ]
            

            save_file(self.data_transformation_config.preprocessing_obj_data_path,preprocessing_obj)

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessing_obj_data_path

            )

        
        except Exception as e:
            raise CustomException(e,sys)







