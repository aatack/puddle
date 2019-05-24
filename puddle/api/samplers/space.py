from puddle.api.sampler import Sampler


class SpaceSampler(Sampler):
    def __init__(self, spaces, valid_equations):
        """Create a sampler that takes samples uniformly from a space."""
        super().__init__(spaces, valid_equations)
        self._setup_equation_weights()
        self._setup_space_lambdas()

    def _setup_equation_weights(self):
        """Create a constant dictionary to pass as the equation weights."""
        weight = 1 / len(self.equations)
        self.equation_weights = {equation: weight for equation in self.equations}

    def _setup_space_lambdas(self):
        """Create a dictionary of functions to sample from each space individually."""
        self.space_lambdas = {
            space: lambda: np.random.uniform(
                low=space.lower, high=space.upper, size=space.shape
            )
            for space in self.independent_variables
        }

    def get_individual_sample(self):
        """Sample the space uniformly."""
        return self._execute_lambdas(), self.equation_weights

    def _execute_lambdas(self):
        """Execute each of the space lambdas in turn."""
        return {
            space: space_lambda() for space, space_lambda in self.space_lambdas.items()
        }
