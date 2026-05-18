import numpy as np
import pandas as pd

import joblib

model=joblib.load(
    "../models/random_forest.pkl"
)

def clean_and_predict(
        apartment_dict
):
    input_df=pd.DataFrame(
        [apartment_dict]
    )
    prediction=model.predict(
        input_df
    )
    final_price=np.expm1(
        prediction[0]
    )
    return final_price