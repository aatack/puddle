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
        return builder.duplicate_for_batch(
            tf.constant(self.wrapped_value), dtype=tf.float32
        )

    def compile(self, compilation_data):
        """Compile a tensorflow node for the variable using the given compiler."""
        return tf.constant(self.wrapped_value)

    def add_compiled_structure(self, structure):
        """Add the compiled structure of the variable to a structure dictionary."""
        if self not in structure:
            structure.add_key(self, tf.float32)

    @staticmethod
    def wrap(value):
        """Wrap the value in a constant variable if it is not already a variable."""
        return value if isinstance(value, Variable) else Constant(value)
