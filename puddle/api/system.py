from puddle.construction.compiler import Compiler
from puddle.api.sampler import Sampler
import tensorflow as tf
import numpy as np


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

    def export(self, variable, arguments=None, unwrap_single_values=True):
        """
        Export a variable to a callable function that calculates it directly.

        If the arguments are left blank and the variable is a dependent variable,
        its default arguments will be used in that order.
        """
        if not self.compiled:
            raise Exception(
                "cannot export variables until the system has been compiled"
            )

        if arguments is None:
            arguments = variable.arguments

        output_node = self.graph.get_outputs(variable)

        def wrap_input(input_tensor):
            single_wrapped = np.array(input_tensor)
            return (
                single_wrapped
                if len(single_wrapped.shape) == len(variable.shape) + 1
                else np.array([single_wrapped])
            )

        def make_feed_dict(fed_arguments):
            return self.graph.get_inputs(
                {
                    arg: wrap_input(fed_arg)
                    for arg, fed_arg in zip(arguments, fed_arguments)
                }
            )

        def calculate(*inputs):
            """Calculate the value of a variable as a function of its arguments."""
            output = self.session.run(output_node, feed_dict=make_feed_dict(inputs))
            if unwrap_single_values and output.shape[0] == 1:
                return output[0]
            else:
                return output

        return calculate
