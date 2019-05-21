from puddle.construction.variable import Variable


def wrap(tensorflow_function, shape_function):
    """Wrap a tensorflow function to be used to transform variables."""

    def inner_wrap(*args, **kwargs):
        shape = shape_function(*args, **kwargs)

        def build_function(variable, builder):
            mapped_list_args = [builder[arg] for arg in args]
            mapped_dict_args = {k: builder[v] for k, v in kwargs.items()}
            return tensorflow_function(*mapped_list_args, **mapped_dict_args)

        return AnonymousVariable(build_function, shape)

    return inner_wrap


class AnonymousVariable(Variable):
    def __init__(self, build_function, shape):
        """Create an anonymous variable from its shape and build function."""
        super().__init__(shape)
        self.build_function = build_function

    def build(self, builder):
        """Build a tensorflow representation of the variable."""
        return self.build_function(self, builder)


class ShapeFunctions:
    @staticmethod
    def copy_first_shape(*args, **kwargs):
        """Copies the shape of the first variable."""
        return args[0].shape
