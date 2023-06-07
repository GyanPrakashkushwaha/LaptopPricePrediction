from src.logger import logging
import sys

def errorMSG(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename

    line_no = exc_tb.tb_lineno

    error_msg = f"Error occured in python file name [{file_name}] line number [{line_no}] error message [{str(error)}]"

    return error_msg

class CustomException(Exception):
    def __init__(self, error_msg,error_detail:sys) -> None:
        self.error_msg = errorMSG(error=error_msg,error_detail=error_detail)

    def __str__(self) -> str:
        return self.error_msg
    

if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as e:
        logging.info("ZERO Division error")
        raise CustomException(error_msg=e,error_detail=sys)
        # raise CustomException(e,sys)
    