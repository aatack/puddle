class Variable:
    def __init__(self, shape, intrinsic_dimension=None):
        """Create a new variable."""
        self.shape = shape
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
    def __init__(self, arguments, layers, shape=[]):
        """Create a dependent variable which is calculated from dependents."""
        super(shape)
        self.arguments = arguments if isinstance(arguments, list) else [arguments]
        self.layers = layers if isinstance(layers, list) else [layers]

    def build(self, builder):
        """Build a tensorflow representation of the variable."""
        raise NotImplementedError()
