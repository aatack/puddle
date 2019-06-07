from puddle.api.sampler import Sampler


class AnonymousSampler:
    def __init__(
        self,
        independent_variables,
        equations,
        single_variable_sample,
        single_equation_sample,
    ):
        """Class for samplers that can be constructed on the fly."""
        super().__init__(independent_variables, equations)
        self.single_variable_sample = single_variable_sample
        self.single_equation_sample = single_equation_sample

    def get_sample(self, size):
        """
        Retrieve a batch of samples from the sampler.

        Each sample should be a tuple of two dictionaries, the first mapping
        independent variables to their values (as numpy arrays) and the second
        mapping equations to their weights (as floats).  All values should be
        given as numpy arrays, where the 0th dimension is the size of the batch.
        """
        variable_samples = [self.single_variable_sample() for _ in range(size)]
        equation_samples = [self.single_equation_sample() for _ in range(size)]
        variable_outputs, equation_outputs = (
            {variable: [] for variable in self.independent_variables},
            {equation: [] for equation in self.equations},
        )
        for sample in variable_samples:
            for variable in variable_outputs:
                if variable in sample:
                    variable_outputs[variable].append(sample[variable])
                else:
                    raise ValueError("no sample provided for {}".format(str(variable)))
        for sample in equation_samples:
            for equation in equation_outputs:
                if equation in sample:
                    equation_outputs[equation].append(sampler[equation])
                else:
                    equation_outputs[equation].append(0.0)
        return variable_outputs, equation_outputs
