from puddle.construction.compiler import Compiler
from puddle.api.sampler import Sampler
import tensorflow as tf


class System:
    def __init__(self, independent_variables, equations):
        """Set up a system of equations with some user-friendly functions exposed."""
        self.compiler = Compiler(independent_variables, equations)
        self.graph = None
        self.session = tf.Session()

        self.sampler_list = []
        self.sampler = None

        self.refresh_sampler()

    def compile(self, initialise=True):
        """Compile a tensorflow graph for the system."""
        self.graph = self.compiler.compile()

        if initialise:
            self.initialise()

        self.run = lambda ins, outs: self.graph.run(self.session, ins, feed_dict=outs)
        return self

    def initialise(self):
        """Initialise variables for the current tensorflow session."""
        self.session.run(tf.global_variables_initializer())

    def refresh_sampler(self):
        """Produce a composite sampler from the currently registered samplers."""
        if len(self.sampler_list) == 0:
            self.sampler = Sampler.placeholder
        else:
            self.sampler = Sampler.composite(self.sampler_list)

    def add_sampler(self, sampler, weight=1.0):
        """Add a sampler to the list of samplers used."""
        self.sampler_list.append((sampler, weight))
        self.refresh_sampler()
