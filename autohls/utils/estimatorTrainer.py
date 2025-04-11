class EstimatorTrainer:
    def __init__(self, filesDict, estimator):
        self._filesDict = filesDict
        self.estimator = estimator
    def trainUntilErrorThreshold(self, data, pragmas, space):
        self.estimator.trainModel(pragmas, data)
