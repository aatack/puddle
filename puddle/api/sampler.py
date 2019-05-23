import numpy as np


class Sampler:
    def __init__(self, independent_variables, losses):
        """Create a new sampler."""
        if not isinstance(independent_variables, set) or not isinstance(losses, set):
            raise ValueError("independent values and losses must be sets")
        self._independent_variables = independent_variables
        self._losses = losses

    def get_independent_variables(self):
        """Return the set of independent variables used by this sampler."""
        return self._independent_variables

    def get_losses(self):
        """Return the set of losses for equations valid under this sampler."""
        return self._losses

    def get_separated_sample(self, size):
        """Get a batch of samples, returning variables and losses separately."""
        batch = self.get_sample(size)
        vars, losses = [], []
        for var, loss in batch:
            vars.append(var)
            losses.append(loss)
        return vars, losses

    def get_sample(self, size):
        """Get a batch of samples."""
        return [self.get_individual_sample() for _ in range(size)]

    def get_individual_sample(self):
        """
        Retrieve a set of samples from the sampler.

        Each sample should be a tuple of two dictionaries, the first mapping
        independent variables to their values (as numpy arrays) and the second
        mapping losses to their weights (as floats).
        """
        raise NotImplementedError()

    @staticmethod
    def space(spaces, valid_equations):
        """Create a samples that takes samples uniformly from spaces."""
        return SpaceSampler(spaces, valid_equations)


class SpaceSampler(Sampler):
    def __init__(self, spaces, valid_equations):
        """Create a sampler that takes samples uniformly from a space."""
        super().__init__(wrap_in_set(spaces), wrap_in_set(valid_equations))
        self._setup_loss_weights()
        self._setup_space_lambdas()

    def _setup_loss_weights(self):
        """Create a constant dictionary to pass as the loss weights."""
        weight = 1 / len(self.get_losses())
        self.loss_weights = {loss: weight for loss in self.get_losses()}

    def _setup_space_lambdas(self):
        """Create a dictionary of functions to sample from each space individually."""
        self.space_lambdas = {
            space: lambda: np.random.uniform(
                low=space.lower, high=space.upper, size=space.shape
            )
            for space in self.get_independent_variables()
        }

    def get_individual_sample(self):
        """Sample the space uniformly."""
        return self._execute_lambdas(), self.loss_weights

    def _execute_lambdas(self):
        """Execute each of the space lambdas in turn."""
        return {
            space: space_lambda() for space, space_lambda in self.space_lambdas.items()
        }


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
