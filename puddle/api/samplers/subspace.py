from puddle.api.sampler import Sampler
import numpy as np


class SubspaceSampler(Sampler):
    def __init__(self, variable, equations, intrinsic_shape):
        """Base class for samplers that sample from a subset of a space."""
        super().__init__(variable, equations)
        self.intrinsic_shape = intrinsic_shape
        self.variable = variable

        self.vectorised_map = lambda l: np.array(
            [self.map_latent_variables(_l) for _l in l]
        )
        self.equation_weight = 1.0 / len(self.equations)

    def _get_latent_variables(self, size):
        """Get a random uniform tensor of the intrinsic shape of the subspace."""
        return np.random.uniform(size=(size,) + self.intrinsic_shape)

    def map_latent_variables(self, latent_tensor):
        """Map the latent variables into the space of the sampler."""
        raise NotImplementedError()

    def get_sample(self, size):
        """
        Retrieve a batch of samples from the sampler.

        Each sample should be a tuple of two dictionaries, the first mapping
        independent variables to their values (as numpy arrays) and the second
        mapping equations to their weights (as floats).  All values should be
        given as numpy arrays, where the 0th dimension is the size of the batch.
        """
        weights = np.repeat(self.equation_weight, size)
        return (
            {self.variable: self.vectorised_map(self._get_latent_variables(size))},
            {equation: weights for equation in self.equations},
        )


class HyperplaneSampler(SubspaceSampler):
    def __init__(self, variable, equations, origin, axes):
        """
        Create a sampler that draws samples from a hyperplane.
        
        The axes should be a matrix with shape (m, n), where n is the number of
        dimensions of the space being sampled, and m is the intrinsic dimensionality
        of the hypeplane being sampled.
        """
        super().__init__(
            variable, equations, (HyperplaneSampler._preformat_axes(axes).shape[0], 1)
        )
        self.origin = origin
        self.axes = HyperplaneSampler._preformat_axes(axes)

        self.space_dimension, self.hyperplane_dimension = self.axes.shape

    def map_latent_variables(self, latent_tensor):
        """Map the latent variables into the space of the sampler."""
        return self.origin + np.sum(self.axes * latent_tensor, axis=0)

    @staticmethod
    def _preformat_axes(axes):
        """Format the axes to be in the form of a rank-2 numpy tensor."""
        if not isinstance(axes, np.ndarray):
            axes = numpy.array(axes)
        while len(axes.shape) < 2:
            axes = np.array([axes])
        return axes
