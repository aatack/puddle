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
        row_derivative_function = self.row_derivative_function(builder)
        return tf.map_fn(
            row_derivative_function, builder.batch_indices, dtype=tf.float32
        )

    def row_derivative_function(self, builder):
        """Create a function that takes a row index and returns its derivatives."""
        variable = builder[self.variable]
        with_respect_to = builder[self.with_respect_to]

        def derivatives_scalar_variable(i):
            return tf.gradients(variable[i], with_respect_to[i])[0]

        def derivatives_vector_variable(i):
            return tf.map_fn(lambda v: tf.gradients(v, with_respect_to[i]), variable[i])

        return (
            derivatives_scalar_variable
            if self.variable.rank == 0
            else derivatives_vector_variable
        )
