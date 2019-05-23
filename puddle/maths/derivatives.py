from puddle.construction.variable import Variable
import tensorflow as tf


class Derivative(Variable):
    def __init__(self, variable, with_respect_to, times=1):
        """Represent the derivative of one variable with respect to another."""
        super().__init__(variable.shape)

        if variable.rank != 1 and variable.rank != 0:
            raise ValueError("variable must be either a vector or scalar")
        if with_respect_to.rank != 0:
            raise ValueError("respective variable must be a scalar")

        self.variable = variable
        self.with_respect_to = with_respect_to
        self.times = times

    def build(self, builder):
        """Build a tensorflow representation of the variable."""
        variable = builder[self.variable]
        with_respect_to = builder[self.with_respect_to]
        if self.variable.rank == 1:
            return tf.stack(
                [
                    tf.gradients(variable[i], with_respect_to)[0]
                    for i in range(self.variable.shape[0])
                ]
            )
        else:
            return tf.gradients(variable, with_respect_to)[0]
