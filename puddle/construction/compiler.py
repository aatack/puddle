from puddle.construction.variable import Variable
from puddle.construction.constant import Constant
import tensorflow as tf


class Compiler:
    def __init__(self, independent_variables, equations):
        """Create a compiler to build a tensorflow graph from a set of variables."""
        self.independent_variables = self.set_wrap(independent_variables)
        self.equations = self.set_wrap(equations)

        self.equation_weight_placeholders = {}

    def compile(self):
        """Compile a tensorflow representation of the system's equations."""
        independent_variable_placeholders = (
            self.build_independent_variable_placeholders()
        )
        equation_weight_placeholders = self.build_equation_weight_placeholders()

        equation_nodes, all_nodes = tf.map_fn(
            self.map_inputs,
            (independent_variable_placeholders, equation_weight_placeholders),
            dtype=self.get_mapped_type(),
        )
        equation_nodes["batch_mean"] = tf.reduce_mean(equation_nodes["mean"])
        equation_nodes["weights"] = equation_weight_placeholders
        return CompiledGraph(
            self, independent_variable_placeholders, equation_nodes, all_nodes
        )

    def map_inputs(self, inputs):
        """Compile one set of the system's equations."""
        independent_variable_placeholders, equation_weight_placeholders = inputs
        compilation_data = CompilationData(independent_variable_placeholders)
        equation_nodes = self.compile_equations(
            compilation_data, equation_weight_placeholders
        )

        return equation_nodes, compilation_data.export_all()

    def compile_equations(self, compilation_data, equation_weight_placeholders):
        """Create weighted nodes for each equation but do not aggregate them."""
        unweighted = {
            equation: compilation_data.get(equation) for equation in self.equations
        }
        weighted = {
            equation: equation_weight_placeholders[equation] * unweighted[equation]
            for equation in self.equations
        }
        return {
            "unweighted": unweighted,
            "weights": equation_weight_placeholders,
            "weighted": weighted,
            "mean": tf.reduce_mean(tf.stack(list(weighted.values()), axis=0)),
        }

    def build_independent_variable_placeholders(self):
        """Get placeholder tensors for each independent variable."""
        return {
            var: self.make_placeholder(var.shape) for var in self.independent_variables
        }

    def build_equation_weight_placeholders(self):
        """Get placeholder tensors for the weight of each equation."""
        return {equation: self.make_placeholder(()) for equation in self.equations}

    def get_mapped_type(self):
        """Return the structure of the tensorflow graph upon compiled."""
        return (self._get_equation_nodes_structure(), self._get_all_nodes_structure())

    def _get_equation_nodes_structure(self):
        """Return the structure of the equation nodes index generated by compilation."""
        equation_types = {equation: tf.float32 for equation in self.equations}
        return {
            "unweighted": equation_types,
            "weights": equation_types,
            "weighted": equation_types,
            "mean": tf.float32,
        }

    def _get_all_nodes_structure(self):
        """Return the structure of the node dump generated upon compilation."""
        structure = CompilationStructure()
        for variable in self.independent_variables:
            structure.set_variable(variable)
        for equation in self.equations:
            structure.set_variable(equation)
        return structure.structure

    def make_placeholder(self, shape, data_type=tf.float32):
        """Make a placeholder node of the given shape."""
        return tf.placeholder(data_type, shape=(None,) + shape)

    @staticmethod
    def set_wrap(values):
        """Wrap the argument in a set if they are not already."""
        if isinstance(values, set):
            return values
        elif isinstance(values, dict):
            return set(values.values())
        elif isinstance(values, list):
            return set(values)
        else:
            return {values}

    @staticmethod
    def join_dictionaries(a, b):
        """Merge the values of two dictionaries, prioritising those in the first."""
        c = {k: v for k, v in b.items()}
        for k, v in a.items():
            c[k] = v
        return c


class CompilationData:
    def __init__(self, placeholders={}):
        """Create a data class for storing tensorflow nodes during compilation."""
        self.instances = {k: v for k, v in placeholders.items()}
        self.flattened_instances = {}

    def get(self, variable):
        """Retrieve the tensorflow node for the given variable."""
        if variable not in self.instances:
            self.instances[variable] = (
                variable.compile(self)
                if isinstance(variable, Variable)
                else Constant.wrap(variable).compile(self)
            )
        return self.instances[variable]

    def flatten(self, variable):
        """Retrieve a flattened version of the variable's tensorflow node."""
        if variable not in self.flattened_instances:
            self.flattened_instances[variable] = tf.reshape(self.get(variable), [-1])
        return self.flattened_instances[variable]

    def join(self, variables):
        """Flatten each of the given variables and concatenate them."""
        return tf.concat([self.flatten(variable) for variable in variables], axis=0)

    def export_all(self):
        """Return a complete dictionary of variables mapped to tensorflow tensors."""
        return self.instances


class CompilationStructure:
    def __init__(self):
        """Create a data class for storing the compiled structure of a system."""
        self.structure = {}

    def set_variable(self, variable):
        """Wrap a variable and ask it to set its structure."""
        if isinstance(variable, Variable):
            variable.add_compiled_structure(self)
        else:
            wrapped = Constant.wrap(variable)
            wrapped.add_compiled_structure(self)
            self.structure[variable] = self.structure[wrapped]
            del self.structure[wrapped]

    def add_key(self, variable, structure):
        """Add a key-value pair to the structure and do nothing else."""
        self.structure[variable] = structure

    def __contains__(self, variable):
        """Determine whether or not the variable is already listed."""
        return variable in self.structure


class CompiledGraph:
    def __init__(self, compiler, variable_nodes, equation_nodes, all_nodes):
        """Create an object for easily accessing compiled nodes."""
        self.compiler = compiler
        self.variable_nodes = variable_nodes
        self.equation_nodes = equation_nodes
        self.all_nodes = all_nodes

    def get_inputs(self, variables):
        """
        For each variable given, return the corresponding input node.
        
        If the variable is an independent variable, its placeholder will be
        returned.  If it is an equation, its weight placeholder will be
        returned.  Otherwise, an error will be thrown.
        """
        return nested_map(self._get_input, variables, map_keys=True, map_values=False)

    def _get_input(self, variable):
        """Retrieve a single input's placeholder node."""
        if variable in self.variable_nodes:
            return self.variable_nodes[variable]
        elif variable in self.equation_nodes["weights"]:
            return self.equation_nodes["weights"][variable]
        else:
            raise ValueError("cannot get the input of an output node")

    def get_outputs(self, variables, weighted_equations=True):
        """For each variable given, return the corresponding output node."""
        getter = self._get_output_function(weighted_equations=weighted_equations)
        return nested_map(getter, variables)

    def _get_output_function(self, weighted_equations=True):
        """Return a function that retrieves output nodes."""
        equations = self.equation_nodes[
            "weighted" if weighted_equations else "unweighted"
        ]

        def _get_output(variable):
            if variable in equations:
                return equations[variable]
            elif variable in self.variable_nodes:
                return self.variable_nodes[variable]
            else:
                return self.all_nodes[variable]

        return _get_output

    def get_mean_losses(self):
        """Return the node for mean weighted loss for each sample in a batch."""
        return self.equation_nodes["mean"]

    def get_batch_mean_loss(self):
        """Return the node for mean weighted loss across all samples in a batch."""
        return self.equation_nodes["batch_mean"]

    def run(self, session, queries, feed_dict={}, weighted_equations=True):
        """Run the graph given some inputs."""
        return session.run(
            self.get_outputs(queries, weighted_equations=weighted_equations),
            feed_dict=self.get_inputs(feed_dict),
        )


def nested_map(f, data, map_values=True, map_keys=False):
    """Map over a data structure, keeping form while changing root values."""
    recall = lambda x: nested_map(f, x, map_keys=map_keys, map_values=map_values)
    recall_value = recall if map_values else lambda v: v
    recall_key = recall if map_keys else lambda k: k
    if isinstance(data, dict):
        return {recall_key(key): recall_value(value) for key, value in data.items()}
    elif isinstance(data, tuple):
        return tuple(recall(datum) for datum in data)
    elif isinstance(data, list):
        return [recall(datum) for datum in data]
    elif isinstance(data, set):
        return {recall(datum) for datum in data}
    else:
        return f(data)
