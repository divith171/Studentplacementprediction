import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from Sampler.OverSampler import Sampling


class DataScaler:

    def __init__(self):
        self.data = Sampling()

    def scaledata(self):
        x, y = self.data.oversampling()
        sclr = StandardScaler()
        x = pd.DataFrame(sclr.fit_transform(x), columns=x.columns)
        return x, y, sclr

    def serializescalar(self):
        a, b, s = self.scaledata()
        pickle.dump(s, open('Scalar.pkl', 'wb'))




