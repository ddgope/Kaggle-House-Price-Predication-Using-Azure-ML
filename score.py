import json
import numpy as np
import os
import joblib
import pandas as pd

def init():
    #This function initialises the model. The model file is retrieved used within the script.
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'best_hyperdrive_model.pkl') #name of model file (.sav or .pkl)
    print("Found model:", os.path.isfile(model_path)) #To check whether the model is actually present on the location we are looking at
    model = joblib.load(model_path)

#===================================================================
#Input the data as json and returns the predictions in json. All preprocessing  steps are specific to this model and usecase
#=================================================================== 
def run(data):
    try:
        #data = np.array(json.loads(data))
        data = json.loads(data)['data'] # raw = pd.read_json(data) 
        data = pd.DataFrame.from_dict(data)

         #prediction steps 
        result = model.predict(data)

        #packaging steps 
        #result = pred.to_json()

        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
        
    except Exception as e:
        error = str(e)
        return error