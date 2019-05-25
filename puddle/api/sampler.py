from random import random


class Sampler:
    def __init__(self, independent_variables, equations):
        """Create a new sampler."""
        self.independent_variables = wrap_in_set(independent_variables)
        self.equations = wrap_in_set(equations)

    def get_separated_sample(self, size):
        """Get a batch of samples, returning variables and equations separately."""
        raise Exception("get_separated_sample has been removed")

    def get_joined_sample(self, size):
        """Retrieve a batch of samples, then put the results all into one dictionary."""
        variable_values, weight_values = self.get_sample(size)
        for weight, value in weight_values.items():
            variable_values[weight] = value
        return variable_values

    def get_sample(self, size):
        """
        Retrieve a batch of samples from the sampler.

        Each sample should be a tuple of two dictionaries, the first mapping
        independent variables to their values (as numpy arrays) and the second
        mapping equations to their weights (as floats).  All values should be
        given as numpy arrays, where the 0th dimension is the size of the batch.
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

    def get_sample(self, size):
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
