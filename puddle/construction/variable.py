from puddle.util.guid import guid
import puddle.util.reusablenet as rnet
import puddle.util.batchless as bl
import tensorflow as tf


class Variable:

    variable_id = 0

    def __init__(self, shape, intrinsic_dimension=None):
        """Create a new variable."""
        self.shape = tuple(shape)
        self.intrinsic_dimension = intrinsic_dimension or self.represented_dimension
        self.rank = len(self.shape)

        self.id = Variable.variable_id
        Variable.variable_id += 1

    def __lt__(self, other):
        """Implement variable sorting so that dictionaries can be flattened."""
        return self.id < other.id if isinstance(other, Variable) else self.id < other

    def __gt__(self, other):
        """Implement variable sorting so that dictionaries can be flattened."""
        return self.id > other.id if isinstance(other, Variable) else self.id > other

    def build(self, builder):
        """Build a tensorflow representation of the variable."""
        raise NotImplementedError()

    def compile(self, compilation_data):
        """Compile a tensorflow node for the variable using the given compiler."""
        raise NotImplementedError()

    def add_compiled_structure(self, structure):
        """Add the compiled structure of the variable to a structure dictionary."""
        raise NotImplementedError()

    @property
    def represented_dimension(self):
        """Calculate the intrinsic dimension of the variable."""
        product = 1
        for dimension in self.shape:
            product *= dimension
        return product


class DeprecatedDependentVariable(Variable):
    def __init__(self, arguments, layers):
        """Create a dependent variable which is calculated from dependents."""
        super().__init__((layers[-1][0],))
        self.arguments = list_wrap(arguments)
        self.layers = list_wrap(layers)

        self.input_dimension = sum(
            [variable.represented_dimension for variable in self.arguments]
        )
        self.network_input_dict = rnet.feedforward_network_input_dict(
            guid(), self.input_dimension, self.layers
        )

    def build(self, builder):
        """Build a tensorflow representation of the variable."""
        copy = rnet.deep_copy(self.network_input_dict)
        copy["name"] = guid()
        copy["input"] = builder.join(*self.arguments)
        return rnet.build_feedforward_network(copy)["output"]


class DependentVariable(Variable):
    def __init__(self, arguments, layers):
        """Create a variable which is a function of one or more dependents."""
        super().__init__(list_wrap(layers)[-1][0])

        self.arguments = list_wrap(arguments)
        self.layers = list_wrap(layers)

        self.apply_to = self.make_application_function()

    def make_application_function(self):
        """Make a function which can be called repeatedly to compile the variable."""
        previous_shape = (
            sum([product(argument.shape) for argument in self.arguments]),
        )
        layers = []
        for output_shape, activation in self.layers:
            layers.append(bl.make_layer(previous_shape, output_shape, activation))
            previous_shape = output_shape
        return bl.compose(*layers)

    def compile(self, compilation_data):
        """Compile a tensorflow node for the variable using the given compiler."""
        return self.apply_to(compilation_data.join(self.arguments))

    def add_compiled_structure(self, structure):
        """Add the compiled structure of the variable to a structure dictionary."""
        if self not in structure:
            structure.add_key(self, tf.float32)
            for argument in self.arguments:
                structure.set_variable(argument)


def product(values):
    """Calculate the product of a list of values."""
    total = 1
    for value in values:
        total *= value
    return total


def list_wrap(value):
    """Wrap the value in a list if it is not already a list."""
    return value if isinstance(value, list) else [value]
