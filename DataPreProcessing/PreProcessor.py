import pandas as pd
import numpy as np
from DataAcquisition.Loader import AcquireData
pd.set_option("display.max_columns",None)


class FeatEng:

    def __init__(self):
        self.data = AcquireData()

    def replace(self):
        data = self.data.access()
        data['ssc_b'] = np.where(data['ssc_b'] == 'Others', 0, 1)
        data['hsc_b'] = np.where(data['hsc_b'] == 'Others', 0, 1)
        data['workex'] = np.where(data['workex'] == 'No', 0, 1)
        data['specialisation'] = np.where(data['specialisation'] == 'Mkt&HR', 0, 1)
        data['status'] = np.where(data['status'] == 'Placed', 1, 0)
        data['gender'] = np.where(data['gender']=='M',1,0)
        return data

    def encoder(self):
        data = self.replace()
        data = pd.get_dummies(data, columns=['hsc_s','degree_t'],drop_first=True)
        return data

    def drop(self):
        data = self.encoder()
        data.drop(['salary', 'sl_no'], axis=1, inplace=True)
        return data


