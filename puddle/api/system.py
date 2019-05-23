from puddle.construction.builder import Builder


class System:
    def __init__(self, builder=None, sampler=None):
        """Set up a system of equations with some user-friendly functions exposed."""
        self._builder = None
        self._sampler = None

        if builder is not None:
            self.set_builder(builder)
        if sampler is not None:
            self.set_sampler(sampler)

    def set_builder(self, builder):
        """Set the builder to be used to creating a tensorflow graph."""
        self._builder = builder

    @property
    def builder(self):
        """Return the builder currently being used."""
        if self._builder is None:
            self._builder = Builder()
        return self._builder

    def set_sampler(self, sampler):
        """Set the sampler to be used for training the system."""
        self._sampler = sampler
        self.builder.compile(sampler.get_independent_variables(), sampler.get_losses())

    @property
    def sampler(self):
        """Return the sampler currently being used."""
        if self._sampler is None:
            raise Exception("no sampler specified")
        return self._sampler

    def fit_batch(self, size):
        """Train the system for one batch."""
        builder = self.builder
        return builder.train_on_batch(
            builder.build_feed_dict(*self.sampler.get_separated_sample(size))
        )

    def fit(self, batches, batch_size):
        """Train the system for the specified number of batches."""
        return [self.fit_batch(batch_size) for _ in range(batches)]
