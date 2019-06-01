from puddle.api.sampler import Sampler
import numpy as np


class CompositeSampler(Sampler):
    def __init__(self, sampler_weights):
        """Create a sampler that draws from other samplers with varying probability."""
        super().__init__(
            *Sampler.aggregate_variables_and_equations(
                [sampler for sampler, _ in sampler_weights]
            )
        )
        self.sampler_data = self._get_sampler_data(sampler_weights)

    def _get_sampler_data(self, sampler_weights):
        """Produce a list of SamplerData objects describing the components."""
        total_weight = self._get_total_weight(sampler_weights)
        cumulative_threshold = 0.0
        sampler_data = []

        for sampler, weight in sampler_weights:
            cumulative_threshold += weight / total_weight
            sampler_data.append(
                SamplerData(
                    sampler,
                    cumulative_threshold,
                    self.independent_variables - sampler.independent_variables,
                    self.equations - sampler.equations,
                )
            )

        # Prevent any issues due to rounding at the last sampler:
        sampler_data[-1].cumulative_threshold += 1.0

        return sampler_data

    def _get_total_weight(self, sampler_weights):
        """Return a version of sampler-weight tuples with the weights summing to 1."""
        return sum([w for s, w in sampler_weights])

    def get_sample(self, size):
        """Retrieve a batch of samples from the sampler."""
        randoms = np.sort(np.random.uniform(size=size))

        variable_samples, equation_samples = [], []
        current_sample_size = 0
        current_sampler_index = 0
        current_random_index = 0

        while current_random_index <= size:
            if current_random_index == size or (
                randoms[current_random_index]
                > self.sampler_data[current_sampler_index].cumulative_threshold
            ):
                if current_sample_size > 0:
                    new_variables, new_equations = self.sampler_data[
                        current_sampler_index
                    ].get_sample(current_sample_size)
                    variable_samples.append(new_variables)
                    equation_samples.append(new_equations)
                current_sample_size = 0
                current_sampler_index += 1
                if current_random_index == size:
                    break
            else:
                current_sample_size += 1
                current_random_index += 1
        return (
            self._concatenate_samples(variable_samples, self.independent_variables),
            self._concatenate_samples(equation_samples, self.equations),
        )

    def _concatenate_samples(self, samples, keys):
        """Given a list of dictionaries, concatenate samples for each key."""
        return {
            key: np.concatenate([sample_batch[key] for sample_batch in samples], axis=0)
            for key in keys
        }


class SamplerData:
    def __init__(
        self, sampler, cumulative_threshold, default_variables, default_equations
    ):
        """Data class for storing information on a component of a composite sampler."""
        self.sampler = sampler
        self.cumulative_threshold = cumulative_threshold
        self.default_variables = default_variables
        self.default_equations = default_equations

    def get_sample(self, size):
        """Take a sample and fill in any missing variables and equations."""
        variable_values, equation_weights = self.sampler.get_sample(size)
        for variable in self.default_variables:
            variable_values[variable] = np.zeros((size,) + variable.shape)
        for equation in self.default_equations:
            equation_weights[equation] = np.zeros((size,))
        return variable_values, equation_weights
