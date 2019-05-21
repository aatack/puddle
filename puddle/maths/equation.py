from puddle.construction.variable import Variable
import tensorflow as tf


class Equation(Variable):
    def __init__(self, lhs, rhs=0.0):
        """Create a loss term in which the two sides should be equal."""
        super().__init__(())
        self.lhs = lhs
        self.rhs = rhs

    def build(self, builder):
        """Build a tensorflow representation of the variable."""
        return tf.losses.mean_squared_error(builder[self.lhs], builder[self.rhs])
