class BuildParameters:
    def __init__(self, batch_size=None, sampler=False, placeholder=False):
        """Create a set of parameters to describe how a variable should be built."""
        self.batch_size = batch_size

        if sampler and placeholder:
            raise ValueError("cannot use both sampler and placeholder; pick one")

        self.sampler = sampler if sampler or placeholder else True
        self.placeholder = not self.sampler
