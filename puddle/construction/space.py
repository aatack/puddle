from puddle.construction.variable import Variable
import tensorflow as tf


class Space(Variable):
    def __init__(self, shape, name=None):
        """Create a new space."""
        super().__init__(shape, is_independent=True)
        self.user_provided_name = name

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

    @property
    def name(self):
        """Return a string representation of the node."""
        if self.user_provided_name is not None:
            return self.user_provided_name
        else:
            return super().name


class Scalar(Space):
    def __init__(self, lower=0.0, upper=1.0, name=None):
        """Create a new scalar with upper and lower bounds."""
        super().__init__(tuple(), name=name)
        self.lower = lower
        self.upper = upper


class Vector(Space):
    def __init__(self, dimensions, lower=0.0, upper=1.0, name=None):
        """Create a new vector with upper and lower bounds."""
        super().__init__((dimensions,), name=name)
        self.dimensions = dimensions
        self.lower = lower
        self.upper = upper
