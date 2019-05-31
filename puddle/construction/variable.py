from puddle.construction.repository import PuddleRepository
from puddle.util.guid import guid
import puddle.util.reusablenet as rnet
import puddle.util.batchless as bl
import tensorflow as tf


class Variable:

    variable_id = 0

    def __init__(
        self, shape, intrinsic_dimension=None, is_independent=False, is_equation=False
    ):
        """Create a new variable."""
        self.shape = tuple(shape)
        self.intrinsic_dimension = intrinsic_dimension or self.represented_dimension
        self.rank = len(self.shape)

        self.is_independent = is_independent
        self.is_equation = is_equation
        PuddleRepository.register_variable(
            self, is_independent=self.is_independent, is_equation=self.is_equation
        )

        self.id = Variable.variable_id
        Variable.variable_id += 1

        self.callable = None

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
        if self not in structure:
            structure.add_key(self, tf.float32)
            for argument in self.arguments:
                structure.set_variable(argument)

    @property
    def represented_dimension(self):
        """Calculate the intrinsic dimension of the variable."""
        product = 1
        for dimension in self.shape:
            product *= dimension
        return product

    def export(self, arguments=None, system=None, unwrap_single_values=True):
        """Export the variable, allowing it to be called like a function."""
        if self.callable is not None:
            return

        _system = system if system is not None else PuddleRepository.most_recent_system
        self.callable = _system.export(
            self, arguments=arguments, unwrap_single_values=unwrap_single_values
        )

    def __call__(self, *arguments):
        """Calculate the value of the variable as a function of its arguments."""
        return self.callable(*arguments)

    @property
    def name(self):
        """Return a string representation of the node."""
        return "var{}".format(self.id)

    @property
    def tex_name(self):
        """Return some TeX to represent the variable."""
        return self.name

    def __str__(self):
        """Return the variable's name."""
        return self.name

    def __getitem__(self, i):
        """Return an indexed variable for the given index."""
        return IndexedVariable(self, i)


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
        super().__init__(wrap_int_in_tuple(list_wrap(layers)[-1][0]))

        self.arguments = list_wrap(arguments)
        self.layers = list_wrap(layers)

        self.apply_to = self.make_application_function()

        self.callable = None

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


class IndexedVariable(Variable):
    def __init__(self, target, index):
        """Create a variable that indexes another variable."""
        super().__init__(IndexedVariable._calculate_shape(target))
        self.target = target
        self.index = index

    @staticmethod
    def _calculate_shape(target):
        """Calculate the shape of this variable and throw an error if it is invalid."""
        if not isinstance(target, Variable):
            raise ValueError("cannot index an object which is not a Variable")
        if len(target.shape) == 0:
            raise ValueError("cannot index a scalar")
        return target.shape[1:]

    def compile(self, compilation_data):
        """Compile a tensorflow node for the variable using the given compiler."""
        return compilation_data.get(self.target)[self.index]


def product(values):
    """Calculate the product of a list of values."""
    total = 1
    for value in values:
        total *= value
    return total


def list_wrap(value):
    """Wrap the value in a list if it is not already a list."""
    return value if isinstance(value, list) else [value]


def wrap_int_in_tuple(value):
    """Wrap a value, expected to be an ineger, in a tuple if it is not already."""
    return (value,) if isinstance(value, int) else value
