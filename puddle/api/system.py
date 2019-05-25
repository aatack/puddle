from puddle.construction.compiler import Compiler
from puddle.api.sampler import Sampler
import tensorflow as tf


class System:
    def __init__(self, independent_variables, equations):
        """Set up a system of equations with some user-friendly functions exposed."""
        self.independent_variables = independent_variables
        self.equations = equations

        self.compiler = Compiler(independent_variables, equations)
        self.graph = None
        self.session = tf.Session()

    @property
    def compiled(self):
        """Determine whether or not the system has been compiled."""
        return self.graph is not None

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
