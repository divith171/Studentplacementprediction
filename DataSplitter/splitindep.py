from sklearn.model_selection import train_test_split
from Scaler.scale import DataScaler


class SplitData:

    def __init__(self):
        self.data = DataScaler()

    def splitthedata(self):
        x, y, sclr = self.data.scaledata()
        x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.30,random_state=75)
        return x_train, x_test, y_train, y_test


