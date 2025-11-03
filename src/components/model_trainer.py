import os,sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split Train and Test Datasets")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,
                                             X_test=X_test,y_test=y_test,
                                             models=models)
            # Get best model score from Dictionary
            best_model_score = max(sorted(model_report.values()))
            # Get best model name from Dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            # Best Model
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No Best Model Found",sys)
            logging.info("Found Best Model on Training and Testing Dataset")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            r2_scores=r2_score(y_test,predicted)
            
            return r2_scores
        except Exception as e:
            raise CustomException(e,sys)
