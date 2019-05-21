from puddle.construction.constant import Constant
from puddle.construction.variable import Variable
import tensorflow as tf


class Builder:
    def __init__(self):
        """Create a builder, for construction puddle graphs."""
        self.built_variables = {}
        self.flattened_variables = {}

    def __getitem__(self, variable):
        """Retrieve a built variable, or build it if it has not been built."""
        if variable not in self.built_variables:
            self.built_variables[variable] = (
                variable.build(self)
                if isinstance(variable, Variable)
                else Constant.wrap(variable)
            )

        return self.built_variables[variable]

    def get_flattened(self, variable):
        """Return a flattened version of the variable."""
        if variable not in self.flattened_variables:
            self.flattened_variables[variable] = flatten(self[variable])
        return self.flattened_variables[variable]

    def run(self, session, outputs, feed_dict=dict()):
        """Run the given objects through a tensorflow session."""
        return session.run(
            [self[output] for output in outputs],
            {self[k]: v for k, v in feed_dict.items()},
        )

    def join(self, *variables):
        """Join the given variables together into a single tensor."""
        return tf.concat([self.get_flattened(variable) for variable in variables], 1)


def flatten(node):
    """Flatten the tensorflow node, excluding the first dimension."""
    return tf.reshape(node, [tf.shape(node)[0], -1])
