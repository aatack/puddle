import tensorflow as tf


class Compiler:
    def __init__(self, independent_variables, equations):
        """Create a compiler to build a tensorflow graph from a set of variables."""
        self.independent_variables = self.set_wrap(independent_variables)
        self.equations = self.set_wrap(equations)

        self.built_variables = {}

        self.equation_weight_placeholders = {}

    def compile(self):
        """Compile a tensorflow representation of the system's equations."""
        independent_variable_placeholders = (
            self.build_independent_variable_placeholders()
        )
        equation_weight_placeholders = self.build_equation_weight_placeholders()

        output = tf.map_fn(
            self.map_inputs,
            (independent_variable_placeholders, equation_weight_placeholders),
            dtype=self.get_mapped_type(),
        )
        output[1]["batch_mean"] = tf.reduce_mean(output[1]["mean"])
        return output

    def map_inputs(self, inputs):
        """Compile one set of the system's equations."""
        independent_variable_placeholders, equation_weight_placeholders = inputs
        compilation_data = CompilationData(self.join_dictionaries(*inputs))
        equation_nodes = self.compile_equations(
            compilation_data, equation_weight_placeholders
        )

        return (
            independent_variable_placeholders,
            equation_nodes,
            compilation_data.export_all(),
        )

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
        pass

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
        elif isinstance(value, list):
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

    def get(self, variable):
        """Retrieve the tensorflow node for the given variable."""
        if variable not in self.instances:
            self.instances[variable] = variable.compile(self)
        return self.instances[variable]

    def export_all(self):
        """Return a complete dictionary of variables mapped to tensorflow tensors."""
        return self.instances
