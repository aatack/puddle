from puddle.construction.variable import Variable
import tensorflow as tf


class Space(Variable):
    def __init__(self, shape):
        """Create a new space."""
        super().__init__(shape, is_independent=True)

    def placeholder(self):
        """Create a tensorflow node that allows values to be fed in."""
        shape = (None,) + self.shape
        return tf.placeholder(tf.float32, shape=shape)

    def build(self, builder):
        """Build a tensorflow representation of the variable."""
        return self.placeholder()

    def compile(self, compilation_data):
        """Compile a tensorflow node for the variable using the given compiler."""
        raise Exception("cannot compile independent variables")

    def add_compiled_structure(self, structure):
        """Add the compiled structure of the variable to a structure dictionary."""
        if self not in structure:
            structure.add_key(self, tf.float32)


class Scalar(Space):
    def __init__(self, lower=0.0, upper=1.0):
        """Create a new scalar with upper and lower bounds."""
        super().__init__(tuple())
        self.lower = lower
        self.upper = upper


class Vector(Space):
    def __init__(self, dimensions, lower=0.0, upper=1.0):
        """Create a new vector with upper and lower bounds."""
        super().__init__((dimensions,))
        self.dimensions = dimensions
        self.lower = lower
        self.upper = upper
