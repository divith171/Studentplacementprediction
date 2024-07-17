from imblearn.over_sampling import RandomOverSampler
from Sampler.split import Split


class Sampling:

    def __init__(self):
        self.data = Split()

    def oversampling(self):
        x, y = self.data.splitdep()
        samp = RandomOverSampler()
        x, y = samp.fit_resample(x, y)
        return x, y

