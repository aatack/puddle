from puddle.construction.constant import Constant
from puddle.construction.variable import Variable


class Builder:
    def __init__(self, batch_size=None, sampler=False, placeholder=False):
        """Create a builder, for construction puddle graphs."""
        self.default = BuildParameters(
            batch_size=batch_size, sampler=sampler, placeholder=placeholder
        )
        self.exceptions = {}

        self.built_variables = {}

    def make_exception(self, variable, build_parameters):
        """Declare that a particular variable should be built in a special way."""
        self.exceptions[variable] = build_parameters

    def get_parameters(self, variable):
        """Get the parameters for a given variable."""
        return (
            self.exceptions[variable] if variable in self.exceptions else self.default
        )

    def __getitem__(self, variable):
        """Retrieve a built variable, or build it if it has not been built."""
        if variable not in self.built_variables:
            self.built_variables[variable] = (
                variable.build(self)
                if isinstance(variable, Variable)
                else Constant.wrap(variable)
            )

        return self.built_variables[variable]

    def run(self, session, outputs, feed_dict=dict()):
        """Run the given objects through a tensorflow session."""
        return session.run(
            [self[output] for output in outputs],
            {self[k]: v for k, v in feed_dict.items()},
        )


class BuildParameters:
    def __init__(self, batch_size=None, sampler=False, placeholder=False):
        """Create a set of parameters to describe how a variable should be built."""
        self.batch_size = batch_size

        if sampler and placeholder:
            raise ValueError("cannot use both sampler and placeholder; pick one")

        self.sampler = sampler if sampler or placeholder else True
        self.placeholder = not self.sampler
