from puddle.construction.constant import Constant
from puddle.construction.variable import Variable
import tensorflow as tf
import numpy as np


class Builder:
    def __init__(self):
        """Create a builder, for construction puddle graphs."""
        self.built_variables = {}
        self.flattened_variables = {}
        self.session = tf.Session()

        self.batch_size_placeholder = None
        self.duplicate_for_batch = None
        self._setup_batch_nodes()

        self.independent_variables = None
        self.independent_variable_defaults = None
        self.loss_weights = None
        self.weighted_losses = None

        self.compiled_losses = None
        self.loss = None
        self.optimiser = None

    def _setup_batch_nodes(self):
        """Set up nodes used to stack constants so they are the right shape."""
        self.batch_size_placeholder = tf.placeholder(tf.int32, shape=())

        def duplicate_for_batch(node):
            """Stack a single node into a batch node."""
            rank = tf.rank(node)
            return tf.tile(
                tf.expand_dims(node, axis=0),
                tf.concat(
                    [[self.batch_size_placeholder], tf.tile([1], [rank])], axis=0
                ),
            )

        self.duplicate_for_batch = duplicate_for_batch

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
        self._setup_independent_variables(independent_variables)
        self._setup_losses(losses)
        self._setup_optimiser(normalise_loss)
        self.initialise()

    def _setup_independent_variables(self, independent_variables):
        """Set up a dictionary of independent variables to be used for training."""
        if isinstance(independent_variables, set):
            self.independent_variables = {
                variable: self[variable] for variable in independent_variables
            }
            self.independent_variable_defaults = {
                variable: np.zeros(variable.shape) for variable in independent_variables
            }
        else:
            raise ValueError("independent variables must be a set")

    def _setup_losses(self, losses):
        """Set up a dictionary of loss weights to prepare for training."""
        self.weighted_losses = []
        if isinstance(losses, set):
            self.loss_weights = {}
            for loss in losses:
                weight = tf.placeholder(tf.float32, shape=(None,))
                self.loss_weights[loss] = weight
                self.weighted_losses.append(weight * tf.reduce_mean(self[loss]))
        else:
            raise ValueError("losses must be a set")

    def _setup_optimiser(self, normalise_losses):
        """Compile losses and set up an optimiser for training."""
        self.compiled_losses = tf.stack(self.weighted_losses)
        self.loss = (
            tf.reduce_mean(self.compiled_losses)
            if normalise_losses
            else tf.reduce_sum(self.compiled_losses)
        )
        try:
            self.optimiser = tf.train.AdamOptimizer().minimize(self.loss)
        except ValueError:
            print("WARNING: no variables to train.")
            self.optimiser = None

    def build_feed_dict(self, independent_variables, loss_weights):
        """Build a feed dictionary for the tensorflow session used by the builder."""
        if len(independent_variables) != len(loss_weights):
            raise ValueError("lists of variables and losses must be the same length")

        feed_dict = {self.batch_size_placeholder: len(independent_variables)}
        for variable, placeholder in self.independent_variables.items():
            feed_dict[placeholder] = []
            for var_set in independent_variables:
                feed_dict[placeholder].append(
                    var_set[variable]
                    if variable in var_set
                    else self.independent_variable_defaults[variable]
                )
        for loss, placeholder in self.loss_weights.items():
            feed_dict[placeholder] = []
            for weight_set in loss_weights:
                feed_dict[placeholder].append(
                    weight_set[loss] if loss in weight_set else 0.0
                )
        return feed_dict

    def initialise(self):
        """Initialise the tensorflow session being used."""
        self.session.run(tf.global_variables_initializer())

    def train_on_batch(self, feed_dict):
        """Perform a round of training on the given batch of examples."""
        _, current_loss = self.session.run(
            [self.optimiser, self.loss], feed_dict=feed_dict
        )
        return current_loss


def flatten(node):
    """Flatten the tensorflow node, excluding the first dimension."""
    return tf.reshape(node, [tf.shape(node)[0], -1])
