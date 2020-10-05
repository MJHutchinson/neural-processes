from torch.utils import data
from torch import randint, int32
import numpy as np


class SpatialDataset:
    """Dataset to load the spatial dataset


    References: https://github.com/EmilienDupont/neural-processes/blob/master/utils.py
    """

    def __init__(self, batch_size, max_num_context, max_num_extra_target,  X, Y):
        """
        Args:
        """
        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._max_num_extra_target = max_num_extra_target
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def generate(self):
        """
        
        """

        num_context = np.random.randint(low=5, high=self._max_num_context)
        num_extra_target = np.random.randint(self._max_num_extra_target)
        num_target = num_context + num_extra_target
        
        # shuffle index
        locations = randint(len(self), size=(self._batch_size*num_target,))
        X_target = self.X[locations, :].reshape(self._batch_size, num_target, 2)
        Y_target = self.Y[locations, :].reshape(self._batch_size, num_target, 1)
        X_context = X_target[:, :num_context, :]
        Y_context = Y_target[:, :num_context, :]

        return X_context, Y_context, X_target, Y_target

