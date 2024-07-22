from tensorflow.keras.models import load_model
import numpy as np
from typing import List
import datetime
import time    
import sys

#GoogleSheet
import pandas as pd
import gspread
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

sheet_error = False


# load a model
model1 = load_model('D:/UTOKNOI-ForeCast/UtokNoi')

# input data shape as list
data_sequence = [2, 2, 2, 2, 2]
# convert to sequence array shape
sequence_arr = np.array(data_sequence).reshape(1, 5, 1)

# model prediction
predict = model1.predict(sequence_arr)
# ceiling up prediction values in range(1, 4)
predict_val = np.ceil(predict.flatten())
# convert numpy arr to integer value

predict_val.astype(np.int16)[0]

def Forecasting(model_folder : str, data_input : List) -> int:
    model = load_model(model_folder)
    data_sequence = np.array(data_input).reshape(1, 5, 1)
    predict = model.predict(data_sequence).flatten()
    return predict[0]
    # return np.ceil(predict).astype(np.int16)[0]

# Forecasting(model_folder='UtokNoi/',
#            data_input=[2, 2, 2, 2, 2])

water = []
def week_forecast(model_folder : str, data_input : List) -> List:
    forecast = []
    try:
        current_sequence = data_input
        for i in range(7):
            next_value = Forecasting(model_folder, current_sequence)
            current_sequence.append(next_value)
            forecast.append(next_value)
            current_sequence = current_sequence[1:]
            #print(current_sequence)
         
        # return forecast
    except:
        print("ForeCast Error")
    else:
        print(forecast)
        if sheet_error is False:
            AddSheet(forecast)
        else: 
            print("Can't Add To Sheet")





#Setup Sheet

try:
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]

    # Authenticate using the service account key file
    credentials = Credentials.from_service_account_file('D:/UTOKNOI-ForeCast/utoknoi-1626d836fe56.json', scopes=scopes)
    gc = gspread.authorize(credentials)

    # Initialize GoogleAuth and authorize GoogleDrive
    gauth = GoogleAuth()
    gauth.credentials = credentials
    drive = GoogleDrive(gauth)

    # Open a Google Sheet by its key
    gs = gc.open_by_key("1INXFsPzfxLMgsl_7Zp1nXO1she_XGEchRBWehWUKWv8")

    # Select a worksheet by its name
    worksheet1 = gs.worksheet('Sheet1')
except:
    sheet_error = True
    print("Sheet Error")
else:
    print("Sheet OK")

def progressbar(it, prefix="", size=60, out=sys.stdout): # Python3.6+
    count = len(it)
    start = time.time() # time estimate start
    def show(j):
        x = int(size*j/count)
        # time estimate calculation and string
        remaining = ((time.time() - start) / j) * (count - j)        
        mins, sec = divmod(remaining, 60) # limited to minutes
        time_str = f"{int(mins):02}:{sec:03.1f}"
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count} Est wait {time_str}", end='\r', file=out, flush=True)
    show(0.1) # avoid div/0 
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)

def AddSheet(water):
    da = datetime.datetime.now()
    print("-----------Add Data-----------")
    for i in progressbar(range(100), "Add Data To Sheet: ", 40):
        time.sleep(0.002) 
    df = pd.DataFrame({'Day': ['1','2','3', '4', '5', '6', '7'], 'Water': water})
    # write to dataframe
    worksheet1.clear()
    set_with_dataframe(worksheet=worksheet1, dataframe=df, include_index=False,
    include_column_header=True, resize=True)
    print("-----------Add Data Success-----------")
    


week_forecast(model_folder='D:/UTOKNOI-ForeCast/UtokNoi', data_input=[1, 3, 1, 2, 3])
