from puddle.construction.constant import Constant
from puddle.construction.variable import Variable
import tensorflow as tf


class Builder:
    def __init__(self):
        """Create a builder, for construction puddle graphs."""
        self.built_variables = {}
        self.flattened_variables = {}
        self.session = tf.Session()

        self.independent_variables = None
        self.loss_weights = None
        self.weighted_losses = None

        self.compiled_losses = None
        self.loss = None
        self.optimiser = None

    def __getitem__(self, variable):
        """Retrieve a built variable, or build it if it has not been built."""
        if variable not in self.built_variables:
            self.built_variables[variable] = (
                variable.build(self)
                if isinstance(variable, Variable)
                else Constant.wrap(variable).build(self)
            )

        return self.built_variables[variable]

    def get_flattened(self, variable):
        """Return a flattened version of the variable."""
        if variable not in self.flattened_variables:
            self.flattened_variables[variable] = flatten(self[variable])
        return self.flattened_variables[variable]

    def run(self, outputs, feed_dict=dict()):
        """Run the given objects through a tensorflow session."""
        return self.session.run(
            [self[output] for output in outputs],
            {self[k]: v for k, v in feed_dict.items()},
        )

    def join(self, *variables):
        """Join the given variables together into a single tensor."""
        return tf.concat([self.get_flattened(variable) for variable in variables], 1)

    def compile(self, independent_variables, losses, normalise_loss=True):
        """Compile the builder to perform training."""
        if isinstance(independent_variables, set):
            self.independent_variables = {
                variable: self[variable] for variable in independent_variables
            }
        else:
            raise ValueError("independent variables must be a set")

        self.weighted_losses = []
        if isinstance(losses, set):
            self.loss_weights = {}
            for loss in losses:
                weight = tf.placeholder(tf.float32, shape=())
                self.loss_weights[loss] = weight
                self.weighted_losses.append(weight * tf.reduce_mean(self[loss]))
        else:
            raise ValueError("losses must be a set")

        self.compiled_losses = tf.stack(self.weighted_losses)
        self.loss = (
            tf.reduce_mean(self.compiled_losses)
            if normalise_loss
            else tf.reduce_sum(self.compiled_losses)
        )
        self.optimiser = tf.train.AdamOptimizer().minimize(self.loss)


def flatten(node):
    """Flatten the tensorflow node, excluding the first dimension."""
    return tf.reshape(node, [tf.shape(node)[0], -1])
