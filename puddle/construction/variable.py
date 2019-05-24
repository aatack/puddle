from puddle.util.guid import guid
import puddle.util.reusablenet as rnet
import puddle.util.batchless as bl


class Variable:
    def __init__(self, shape, intrinsic_dimension=None):
        """Create a new variable."""
        self.shape = tuple(shape)
        self.intrinsic_dimension = intrinsic_dimension or self.represented_dimension
        self.rank = len(self.shape)

    def build(self, builder):
        """Build a tensorflow representation of the variable."""
        raise NotImplementedError()

    def compile(self, compilation_data):
        """Compile a tensorflow node for the variable using the given compiler."""
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
        self.arguments = arguments if isinstance(arguments, list) else [arguments]
        self.layers = layers if isinstance(layers, list) else [layers]

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
        super().__init__((layers[-1][0],))
        self.arguments = arguments
        self.layers = layers

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
