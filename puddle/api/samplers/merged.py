from puddle.api.sampler import Sampler


class MergedSampler(Sampler):
    def __init__(self, *samplers):
        """Create a sampler that merges the output of multiple sub-samplers."""
        super().__init__(*Sampler.aggregate_variables_and_equations(samplers))
        # Reverse list so the first samplers have highest priority
        self.samplers = samplers[::-1]

    def get_sample(self, size):
        """
        Retrieve a batch of samples from the sampler.

        Each sample should be a tuple of two dictionaries, the first mapping
        independent variables to their values (as numpy arrays) and the second
        mapping equations to their weights (as floats).  All values should be
        given as numpy arrays, where the 0th dimension is the size of the batch.
        """
        samples = [sampler.get_sample(size) for sampler in self.samplers]
        merged = {}
        for sample in samples:
            for k, v in sample.items():
                merged[k] = v
        return merged
