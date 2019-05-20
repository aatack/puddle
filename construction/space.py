import tensorflow as tf


class Space:
    def sampler(self):
        """Create a tensorflow node that samples from the space."""
        raise NotImplementedError()


class Scalar:
    def __init__(self, lower=0.0, upper=1.0):
        """Create a new scalar with upper and lower bounds."""
        self.lower = lower
        self.upper = upper

    def sampler(self):
        """Sample uniformly from the bounded scalar."""
        return tf.random.uniform((), minval=self.lower, maxval=self.upper)


class Vector:
    def __init__(self, dimensions, lower=0.0, upper=1.0):
        """Create a new vector with upper and lower bounds."""
        self.dimensions = dimensions
        self.lower = lower
        self.upper = upper

    def sampler(self):
        """Sample uniformly from the bounded hypercube."""
        return tf.random.uniform(
            (self.dimensions,), minval=self.lower, maxval=self.upper
        )
