from puddle.api.sampler import Sampler
import numpy as np


class SpaceSampler(Sampler):
    def __init__(self, spaces, valid_equations):
        """Create a sampler that takes samples uniformly from a space."""
        super().__init__(spaces, valid_equations)
        self._setup_equation_weights()
        self._setup_space_lambdas()

    def _setup_equation_weights(self):
        """Create a constant dictionary to pass as the equation weights."""
        self.normalised_weight = 1.0 / len(self.equations)

    def _setup_space_lambdas(self):
        """Create a dictionary of functions to sample from each space individually."""
        self.space_lambdas = {
            space: lambda s: np.random.uniform(
                low=space.lower, high=space.upper, size=(s,) + space.shape
            )
            for space in self.independent_variables
        }

    def get_sample(self, size):
        """Sample the space uniformly."""
        return self._execute_lambdas(size), self._get_equation_weights(size)

    def _execute_lambdas(self, size):
        """Execute each of the space lambdas in turn."""
        return {
            space: space_lambda(size)
            for space, space_lambda in self.space_lambdas.items()
        }

    def _get_equation_weights(self, size):
        """Get vectors describing the weight of each equation for a batch."""
        repeated_weights = np.repeat(self.normalised_weight, size)
        return {equation: repeated_weights for equation in self.equations}
