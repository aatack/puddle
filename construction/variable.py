class Variable:
    def __init__(self, shape):
        """Create a new variable."""
        self.shape = shape

    def build(self, sampling_information):
        """Build a tensorflow representation of the variable."""
        raise NotImplementedError()


class IndependentVariable(Variable):
    def __init__(self, shape=[], lower=0.0, upper=1.0):
        """Create a new independent variable."""
        super(shape)


class DependentVariable(Variable):
    def __init__(self, arguments, layers, shape=[]):
        """Create a dependent variable which is calculated from dependents."""
        super(shape)
        self.arguments = arguments if isinstance(arguments, list) else [arguments]
        self.layers = layers if isinstance(layers, list) else [layers]
