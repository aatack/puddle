from puddle.api.sampler import Sampler


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
        aggregate_variables, aggregate_equations = set(), set()
        for sampler in samplers:
            aggregate_variables = aggregate_variables | sampler.independent_variables
            aggregate_equations = aggregate_equations | sampler.equations
        return aggregate_variables, aggregate_equations

    def get_individual_sample(self):
        """Retrieve a sample from the sampler."""
        seed = random()
        for sampler, weight in self.sampler_weights:
            seed -= weight
            if seed < 0:
                return sampler.get_individual_sample()
        return self.sampler_weights[-1][0].get_individual_sample()
