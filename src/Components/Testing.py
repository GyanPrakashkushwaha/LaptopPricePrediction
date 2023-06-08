from src.Components.Data_Collection import DataCollection
from src.Components.Data_Transformation import DataTransformation


obj = DataCollection()

train_df ,test_df=obj.initiate_data_collection()

data_trans = DataTransformation(train_data_path=train_df,test_data_path=test_df)

data_trans.initiate_data_transformation()
