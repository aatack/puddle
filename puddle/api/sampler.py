from random import random
import numpy as np


class Sampler:
    def __init__(self, independent_variables, equations):
        """Create a new sampler."""
        self.independent_variables = wrap_in_set(independent_variables)
        self.equations = wrap_in_set(equations)

    def get_separated_sample(self, size):
        """Get a batch of samples, returning variables and equations separately."""
        raise Exception("get_separated_sample has been removed")

    def get_sample(self, size):
        """Get a batch of samples."""
        return [self.get_individual_sample() for _ in range(size)]

    def get_individual_sample(self):
        """
        Retrieve a sample from the sampler.

        The sample should be a tuple of two dictionaries, the first mapping
        independent variables to their values (as numpy arrays) and the second
        mapping equations to their weights (as floats).
        """
        raise NotImplementedError()

    @property
    def placeholder():
        """Create and return a placeholder sampler."""
        return PlaceholderSampler()


class PlaceholderSampler(Sampler):
    def __init__(self):
        """Create a placeholder sampler which throws an error when sampled."""
        super().__init__({}, {})

    def get_individual_sample(self):
        """Warn the user that a placeholder sampler is selected."""
        raise Exception("a placeholder sampler is currently in use: please provide one")


def wrap_in_set(values):
    """Wrap the values in a set if they are not already."""
    if isinstance(values, set):
        return values
    elif isinstance(values, list):
        return set(values)
    elif isinstance(values, dict):
        return set(values.values())
    else:
        return {values}
