from DataSplitter.splitindep import SplitData
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier


class Model:

    def __init__(self):
        self.data = SplitData()

    def etc(self):
        x_train, x_test, y_train, y_test = self.data.splitthedata()
        etc = ExtraTreesClassifier()
        etc.fit(x_train, y_train)
        # print(etc.score(x_train, y_train))
        # print(etc.score(x_test, y_test))
        return etc

    def rfc(self):
        x_train, x_test, y_train, y_test = self.data.splitthedata()
        rfc = RandomForestClassifier()
        rfc.fit(x_train, y_train)
        # print(rfc.score(x_train, y_train))
        # print(rfc.score(x_test, y_test))
        return rfc




