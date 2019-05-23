from puddle.construction.variable import Variable
import tensorflow as tf
import numpy as np


class Constant(Variable):
    def __init__(self, value):
        """Create a constant valued variable."""
        super().__init__(Constant.numpy_wrap(value).shape)
        self.value = value
        self.wrapped_value = Constant.numpy_wrap(self.value)

    @staticmethod
    def numpy_wrap(value):
        """Wrap the value in a numpy array."""
        return np.array(value, dtype=np.float32)

    def build(self, builder):
        """Build a tensorflow representation of the variable."""
        return tf.constant(self.wrapped_value)

    @staticmethod
    def wrap(value):
        """Wrap the value in a constant variable if it is not already a variable."""
        return value if isinstance(value, Variable) else Constant(value)
