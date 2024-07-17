from DataPreProcessing.PreProcessor import FeatEng


class Split:

    def __init__(self):
        self.data = FeatEng()

    def splitdep(self):
        data = self.data.drop()
        x = data.drop('status',axis=1)
        y = data['status']
        return x, y