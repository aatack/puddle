from puddle.construction.compiler import Compiler
from puddle.api.sampler import Sampler


class System:
    def __init__(self, independent_variables, equations):
        """Set up a system of equations with some user-friendly functions exposed."""
        self.compiler = Compiler(independent_variables, equations)

        self.sampler_list = []
        self.sampler = Sampler.placeholder

    def compile(self):
        """Compile a tensorflow graph for the system."""
        self.compiler.compile()

    def refresh_sampler(self):
        """Produce a composite sampler from the currently registered samplers."""
        if len(self.sampler_list) == 0:
            self.sampler = Sampler.placeholder
        else:
            self.sampler = Sampler.composite(self.sampler_list)
