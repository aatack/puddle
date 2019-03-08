class Model:
    """Defines a group of nodes and the interactions between them."""

    def __init__(self, name):
        """Create a new model with a certain name."""
        self.name = name
        self.components = {}
        self.links = []

    def reference(self, index):
        """Index a model or one of its components."""
        pass

    def build(self):
        """Build the model."""
        pass

    def ready(self):
        """Check the model is ready for building."""
        pass

    def fork(self, new_name):
        """Copy a fork of the model."""
        pass


class Instruction:
    pass
