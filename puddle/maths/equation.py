from puddle.construction.variable import Variable
import tensorflow as tf


class Equation(Variable):
    def __init__(self, lhs, rhs=0.0):
        """Create an error term in which the two sides should be equal."""
        super().__init__((), is_equation=True)
        self.lhs = lhs
        self.rhs = rhs

    def build(self, builder):
        """Build a tensorflow representation of the variable."""
        difference = builder[self.lhs] - builder[self.rhs]
        return tf.reduce_mean(tf.square(difference))

    def compile(self, compilation_data):
        """Compile a tensorflow node for the variable using the given compiler."""
        difference = compilation_data.get(self.lhs) - compilation_data.get(self.rhs)
        return tf.reduce_mean(tf.square(difference))

    def add_compiled_structure(self, structure):
        """Add the compiled structure of the variable to a structure dictionary."""
        if self not in structure:
            structure.add_key(self, tf.float32)
            structure.set_variable(self.lhs)
            structure.set_variable(self.rhs)
