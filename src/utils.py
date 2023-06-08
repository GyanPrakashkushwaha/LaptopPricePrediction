import pickle
import sys
from src.exception import CustomException
from src.logger import logging


class MainUtils:
    def __init__(self):
        pass
    

    @staticmethod
    def save_obj(file_path,obj:object):

        logging.info('Saving objects through save object function')

        try:
            with open(file_path,'wb') as file_obj:
                pickle.dump(obj,file_obj)

            logging.info(f'saved {file_obj} file')
        
        except Exception as e :
            raise CustomException(e,sys)
        

