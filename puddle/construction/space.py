from puddle.construction.variable import Variable
import tensorflow as tf


class Space(Variable):
    def __init__(self, shape):
        """Create a new space."""
        super().__init__(shape)

    def sampler(self, batch_size=None):
        """Create a tensorflow node that samples from the space."""
        raise NotImplementedError()

    def placeholder(self, batch_size=None):
        """
        Create a tensorflow node that allows values to be fed in.

        To exclude batches, put set the batch size to None.  For a variable batch
        size, set it to 0.
        """
        shape = (
            self.shape
            if batch_size is None
            else (None,) + tuple(self.shape)
            if batch_size <= 0
            else (batch_size,) + tuple(self.shape)
        )
        return tf.placeholder(tf.float32, shape=shape)


class Scalar(Space):
    def __init__(self, lower=0.0, upper=1.0):
        """Create a new scalar with upper and lower bounds."""
        super().__init__(tuple())
        self.lower = lower
        self.upper = upper

    def sampler(self, batch_size=None):
        """Sample uniformly from the bounded scalar."""
        shape = () if batch_size is None else (batch_size,)
        return tf.random.uniform(shape, minval=self.lower, maxval=self.upper)


class Vector(Space):
    def __init__(self, dimensions, lower=0.0, upper=1.0):
        """Create a new vector with upper and lower bounds."""
        super().__init__((dimensions,))
        self.dimensions = dimensions
        self.lower = lower
        self.upper = upper

    def sampler(self, batch_size=None):
        """Sample uniformly from the bounded hypercube."""
        shape = (
            (self.dimensions,) if batch_size is None else (batch_size, self.dimensions)
        )
        return tf.random.uniform(shape, minval=self.lower, maxval=self.upper)
