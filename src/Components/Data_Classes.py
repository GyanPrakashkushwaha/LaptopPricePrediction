from dataclasses import dataclass
import os


@dataclass
class DataCollectionConfig:
    def __init__(self):
        os.makedirs('Data-WareHouse',exist_ok=True)
        
    raw_data_path = os.path.join('Data-WareHouse','data.csv')
    train_data_path = os.path.join('Data-WareHouse','trainData.csv')
    test_data_path = os.path.join('Data-WareHouse','testData.csv')


@dataclass
class DataTransformationConfig:
    data_transformation_dir=os.path.join('Data-WareHouse','data_transformation')
    transformed_train_file_path=os.path.join(data_transformation_dir, 'train.npy')
    transformed_test_file_path=os.path.join(data_transformation_dir, 'test.npy') 
    transformed_object_file_path=os.path.join( data_transformation_dir, 'preprocessing.pkl' )

