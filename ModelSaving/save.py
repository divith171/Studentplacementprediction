import pickle
from ModelTuner.tuner import TuningModel


class SaveModel:

    def __init__(self):
        self.model = TuningModel()
        self.path = "model.pkl"
        self.mode = "wb"

    def save(self):
        model1 = self.model.etctuner()
        model2 = self.model.rfctuner()
        path = self.path
        mode = self.mode
        pickle.dump(model1, open(path, mode))
        pickle.dump(model2, open('model2.pkl', mode))


