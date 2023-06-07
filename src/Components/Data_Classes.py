from dataclasses import dataclass
import os


@dataclass
class DataCollectionConfig:
    def __init__(self):
        os.makedirs('Data-WareHouse',exist_ok=True)
        
    raw_data_path = os.path.join('Data-WareHouse','data.csv')
    train_data_path = os.path.join('Data-WareHouse','trainData.csv')
    test_data_path = os.path.join('Data-WareHouse','testData.csv')


