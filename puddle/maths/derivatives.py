from puddle.construction.variable import Variable
import tensorflow as tf


class Derivative(Variable):
    def __init__(self, target, variable, times=1):
        """Represent the derivative of a target with respect to a variable."""
        super().__init__(target.shape)

        if target.rank != 1 and target.rank != 0:
            raise ValueError("target must be either a vector or scalar")
        if variable.rank != 0:
            raise ValueError("variable must be a scalar")

        self.target = target
        self.variable = variable
        self.times = times

    def build(self, builder):
        """Build a tensorflow representation of the variable."""
        target = builder[self.target]
        variable = builder[self.variable]
        if self.target.rank == 1:
            return tf.stack(
                [
                    tf.gradients(target[i], variable)[0]
                    for i in range(self.target.shape[0])
                ]
            )
        else:
            return tf.gradients(target, variable)[0]
