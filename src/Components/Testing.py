from src.Components.Data_Collection import DataCollection
from src.Components.Data_Transformation import DataTransformation
from src.Components.Model_training import ModelTrainer
import warnings
warnings.filterwarnings('ignore')



obj = DataCollection()
obj.initiate_data_collection()

# train_df ,test_df=obj.initiate_data_collection()

# data_trans = DataTransformation(train_data_path=train_df,test_data_path=test_df)


# train_arr , test_arr ,_=data_trans.initiate_data_transformation()

# train_model = ModelTrainer(train_arr=train_arr,test_arr=test_arr)

# train_model.initiate_model_training()