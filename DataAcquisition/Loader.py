import pandas as pd


class AcquireData:

    def __init__(self):
        self.path = r'../Data/Placement_Data_Full_Class.csv'

    def access(self):
        data = pd.read_csv(self.path)
        return data


