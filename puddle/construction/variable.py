from puddle.util.guid import guid
import puddle.util.reusablenet as rnet


class Variable:
    def __init__(self, shape, intrinsic_dimension=None):
        """Create a new variable."""
        self.shape = tuple(shape)
        self.intrinsic_dimension = intrinsic_dimension or self.represented_dimension
        self.rank = len(self.shape)

    def build(self, builder):
        """Build a tensorflow representation of the variable."""
        raise NotImplementedError()

    @property
    def represented_dimension(self):
        """Calculate the intrinsic dimension of the variable."""
        product = 1
        for dimension in self.shape:
            product *= dimension
        return product


class DependentVariable(Variable):
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
