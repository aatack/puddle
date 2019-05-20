class Variable:
    def __init__(self, shape):
        """Create a new variable."""
        self.shape = shape

    def build(self, auto_sample, exceptions=set()):
        """Build a tensorflow representation of the variable."""
        raise NotImplementedError()

    @property
    def intrinsic_dimension(self):
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
