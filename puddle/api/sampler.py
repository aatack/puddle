from random import random
import numpy as np


class Sampler:
    def __init__(self, independent_variables, equations):
        """Create a new sampler."""
        if not isinstance(independent_variables, set) or not isinstance(equations, set):
            raise ValueError("independent values and equations must be sets")
        self._independent_variables = independent_variables
        self._equations = equations

    def get_independent_variables(self):
        """Return the set of independent variables used by this sampler."""
        return self._independent_variables

    def get_equations(self):
        """Return the set of losses for equations valid under this sampler."""
        return self._equations

    def get_separated_sample(self, size):
        """Get a batch of samples, returning variables and equations separately."""
        batch = self.get_sample(size)
        vars, equations = [], []
        for var, equation in batch:
            vars.append(var)
            equations.append(equation)
        return vars, equations

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

    @staticmethod
    def space(spaces, valid_equations):
        """Create a samples that takes samples uniformly from spaces."""
        return SpaceSampler(spaces, valid_equations)

    @property
    def placeholder():
        """Create and return a placeholder sampler."""
        return PlaceholderSampler()

    @property
    def composite(sampler_weights):
        """Create and return a composite sampler."""
        return CompositeSampler(
            [
                sampler_weight
                if isinstance(sampler_weight, tuple)
                else (sampler_weight, 1.0)
                for sampler_weight in sampler_weights
            ]
        )


class SpaceSampler(Sampler):
    def __init__(self, spaces, valid_equations):
        """Create a sampler that takes samples uniformly from a space."""
        super().__init__(wrap_in_set(spaces), wrap_in_set(valid_equations))
        self._setup_equation_weights()
        self._setup_space_lambdas()

    def _setup_equation_weights(self):
        """Create a constant dictionary to pass as the equation weights."""
        weight = 1 / len(self.get_equations())
        self.equation_weights = {equation: weight for equation in self.get_equations()}

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
        return self._execute_lambdas(), self.equation_weights

    def _execute_lambdas(self):
        """Execute each of the space lambdas in turn."""
        return {
            space: space_lambda() for space, space_lambda in self.space_lambdas.items()
        }


class PlaceholderSampler(Sampler):
    def __init__(self):
        """Create a placeholder sampler which throws an error when sampled."""
        super().__init__({}, {})

    def get_individual_sample(self):
        """Warn the user that a placeholder sampler is selected."""
        raise Exception("a placeholder sampler is currently in use: please provide one")


class CompositeSampler(Sampler):
    def __init__(self, sampler_weights):
        """Create a sampler that draws from other samplers with varying probability."""
        super().__init__(
            *CompositeSampler.aggregate_variables_and_equations(
                [sampler for sampler, _ in sampler_weights]
            )
        )
        self.sampler_weights = self._normalise_weights(sampler_weights)

    def _normalise_weights(self, sampler_weights):
        """Return a version of sampler-weight tuples with the weights summing to 1."""
        total = sum([w for s, w in sampler_weights])
        return [(sampler, weight / total) for sampler, weight in sampler_weights]

    @staticmethod
    def aggregate_variables_and_equations(samplers):
        """Return a summary of all variables and equations in a list of samplers."""
        aggregate_variables, aggregate_equations = {}, {}
        for sampler in samplers:
            aggregate_variables = aggregate_variables | sampler._independent_variables
            aggregate_equations = aggregate_equations | sampler._equations
        return aggregate_variables, aggregate_equations

    def get_individual_sample(self):
        """Retrieve a sample from the sampler."""
        seed = random()
        for sampler, weight in self.sampler_weights:
            seed -= weight
            if seed < 0:
                return sampler.get_individual_sample()
        return self.sampler_weights[-1][0].get_individual_sample()


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
