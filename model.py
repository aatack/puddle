class Model:
    """Defines a group of nodes and the interactions between them."""

    def __init__(self, name):
        """Create a new model with a certain name."""
        self.name = name
        self.components = {}
        self.links = []
        self.instructions = []

    def reference(self, index):
        """Index a model or one of its components."""
        if index is None or \
            (isinstance(index, str) and len(index) == 0) or \
            (isinstance(index, list) and len(index) == 0):
            return self
        else:
            split_index = index if isinstance(index, list) \
                else index.split('.')
            if split_index[0] in self.components:
                return self.components[split_index[0]].reference(split_index[1:])
            else:
                raise KeyError('component "{}" does not exist'.format(split_index[0]))
        
    def build(self):
        """Build the model."""
        pass

    def ready(self):
        """Check the model is ready for building."""
        for component in self.components.values():
            if not component.is_defined():
                return False
            elif not component.ready():
                return False
        return True

    def fork(self, new_name):
        """Copy a fork of the model."""
        pass

    def is_defined(self):
        """Determine whether or not the model is fully defined."""
        return False


class Instruction:
    pass
