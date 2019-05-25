from puddle.api.sampler import Sampler
from puddle.api.samplers.space import SpaceSampler
from puddle.api.samplers.composite import CompositeSampler
import tensorflow as tf


class Trainer:
    def __init__(
        self,
        system,
        samplers=None,
        optimiser=None,
        batch_size=None,
        pre_batch_callbacks=None,
        post_batch_callbacks=None,
    ):
        """Create a trainer which handles the fitting of a system of equations."""
        self.system = system
        self.sampler_list = samplers if samplers is not None else self.default_samplers
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
        self.batch_number = 0

        self.error = None
        self.optimise_op = None

        self.sampler = None
        self.refresh_sampler()

    @property
    def default_samplers(self):
        """Return a sampler that takes uniformly from the space of variables."""
        return [
            (
                SpaceSampler(self.system.independent_variables, self.system.equations),
                1.0,
            )
        ]

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
        return []

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
