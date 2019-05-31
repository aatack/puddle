from puddle.api.system import System
from puddle.api.sampler import Sampler
from puddle.api.samplers.space import SpaceSampler
from puddle.api.samplers.composite import CompositeSampler
import tensorflow as tf


class Trainer:
    def __init__(
        self,
        system=None,
        samplers=None,
        optimiser=None,
        batch_size=None,
        pre_batch_callbacks=None,
        post_batch_callbacks=None,
    ):
        """Create a trainer which handles the fitting of a system of equations."""
        self.batch_number = 0
        self.epoch = None

        self.queries = {}

        self.error = None
        self.optimise_op = None

        self.sampler = None

        self.system = system if system is not None else System()
        self.sampler_list = samplers if samplers is not None else []
        self.optimiser = optimiser if optimiser is not None else self.default_optimiser
        self.batch_size = batch_size or 32

        self.pre_batch_callbacks = (
            pre_batch_callbacks
            if pre_batch_callbacks is not None
            else self.default_pre_batch_callbacks
        )
        self.post_batch_callbacks = (
            post_batch_callbacks
            if post_batch_callbacks is not None
            else self.default_post_batch_callbacks
        )

        self.refresh_sampler()

    @property
    def default_optimiser(self):
        """Return a default optimiser - Adam with default parameters."""
        return tf.train.AdamOptimizer()

    @property
    def default_pre_batch_callbacks(self):
        """Return a list of default pre-batch callbacks."""
        return []

    @property
    def default_post_batch_callbacks(self):
        """Return a list of default post-batch callbacks."""
        self.add_query("epoch", "error")

        def log_epoch(trainer, data):
            if data["epoch"] % 100 == 0:
                print("Epoch: {}\tError: {}".format(data["epoch"], data["error"]))

        return [log_epoch]

    def refresh_sampler(self):
        """Produce a composite sampler from the currently registered samplers."""
        if len(self.sampler_list) == 0:
            self.sampler = Sampler.placeholder
        else:
            self.sampler = CompositeSampler(self.sampler_list)

    def add_sampler(self, sampler, weight=1.0):
        """Add a sampler to the list of samplers used."""
        self.sampler_list.append((sampler, weight))
        self.refresh_sampler()

    def initialise_training(self):
        """Check that everything is in place for training to begin."""
        if not self.system.compiled:
            self.system.compile()

        if self.optimise_op is None:
            self.error = self.system.graph.get_batch_mean_loss()
            self.optimise_op = self.optimiser.minimize(self.error)
            self.system.session.run(
                tf.variables_initializer(self.optimiser.variables())
            )

    def train(self, iterations):
        """Train the system for the specified number of iterations."""
        self.initialise_training()
        epochs = []

        try:
            for _ in range(iterations):
                queries = self.perform_training_iteration()
                epochs.append(queries)
        except KeyboardInterrupt:
            pass

        return epochs

    def perform_training_iteration(self):
        """Train the system on one batch, including triggering all events."""
        sample = self.sampler.get_joined_sample(self.batch_size)
        self._trigger_pre_batch_events(sample)
        queries = self._train_on_batch(sample)
        self._trigger_post_batch_events(queries)
        self.batch_number += 1
        return queries

    def _trigger_pre_batch_events(self, sample):
        """Call all callbacks after getting a sample but before training on it."""
        if len(self.pre_batch_callbacks) > 0:
            for callback in self.pre_batch_callbacks:
                callback(self, sample)

    def _trigger_post_batch_events(self, queries):
        """Call all callbacks on the results of the given queries."""
        if len(self.post_batch_callbacks) > 0:
            for callback in self.post_batch_callbacks:
                callback(self, queries)

    def _train_on_batch(self, sample):
        """Optimise network parameters on the given sample."""
        feed_dict = self.system.graph.get_inputs(sample)
        feed_dict[self._get_epoch_node()] = self.batch_number

        queries, _ = self.system.session.run(
            (self.queries, self.optimise_op), feed_dict
        )
        return queries

    def add_query(self, *variables):
        """
        Add a query for a variable whose value will be returned during training.

        To get the value of a variable, simply pass in the variable.  To get
        the mean error for each sample in the batch, pass in "mean_error", or
        to get overall error simply type "error".  "epoch" gives the current
        training epoch.
        """
        self.initialise_training()
        for variable in variables:
            if variable in self.queries:
                continue

            if isinstance(variable, str):
                options = self._get_string_query_options()
                if variable in options:
                    self.queries[variable] = options[variable]()
                else:
                    raise ValueError("variable '{}' unknown".format(variable))
            else:
                self.queries[variable] = self.system.graph.get_output(variable)

    def _get_string_query_options(self):
        """Get a list of options which can be used to add non-variable queries."""
        return {
            "mean_error": lambda: self.system.graph.get_mean_losses(),
            "error": lambda: self.system.graph.get_batch_mean_loss(),
            "epoch": lambda: self._get_epoch_node(),
        }

    def _get_epoch_node(self):
        """Get a node which can be used to represent the current epoch."""
        if self.epoch is None:
            self.epoch = tf.placeholder(tf.int32, shape=())
        return self.epoch
