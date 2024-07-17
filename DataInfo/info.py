import pandas as pd
import numpy as np
from DataAcquisition.Loader import AcquireData


class Overview:

    def __init__(self):
        self.data = AcquireData()

    def shape(self):
        data = self.data.access()
        return data.shape

    def size(self):
        data = self.data.access()
        return data.size

    def info(self):
        data = self.data.access()
        return data.info()


