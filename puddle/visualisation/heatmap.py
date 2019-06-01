from puddle.construction.variable import Variable
from puddle.construction.space import Space
from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np


class HeatMap:
    def __init__(self, x, y, z, use_tex=False, fidelity=100):
        """Create an object for plotting a heat map of a value."""
        if not all([isinstance(v, Variable) for v in [x, y, z]]):
            raise ValueError("all arguments must be instances of Variable")
        if not isinstance(x, Space) or not isinstance(y, Space):
            raise ValueError("both x and y values must be Spaces")
        if isinstance(z, Space) or len(z.shape) > 0:
            raise ValueError("z must be both a scalar and not a Space")

        self.fidelity = fidelity

        self.x = x
        self.x_range = (x.lower, x.upper)
        self.xs = np.linspace(*self.x_range, self.fidelity)

        self.y = y
        self.y_range = (y.lower, y.upper)
        self.ys = np.linspace(*self.y_range, self.fidelity)

        self.z = z
        self.z.export([self.x, self.y])

        mesh = np.meshgrid(self.xs, self.ys)
        self.x_inputs = np.reshape(mesh[0], [-1])
        self.y_inputs = np.reshape(mesh[1], [-1])

        self.use_tex = use_tex
        if self.use_tex:
            rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
            rc("text", usetex=True)

        self.figure = None
        self.image = None

    def _get_z_data(self):
        """Recalculate the values and return them."""
        return np.reshape(
            self.z(self.x_inputs, self.y_inputs), [self.fidelity, self.fidelity]
        )

    def ready(self):
        """Prepare the heat map for displaying."""
        # plt.ion()
        self.figure = plt.figure()
        plt.ion()

        # if self.y_range is not None:
        #     plt.ylim(*self.y_range)
        # plt.xlim(self.x.lower, self.x.upper)

        plt.xlabel(self.x.tex_name if self.use_tex else self.x.name)
        plt.ylabel(self.y.tex_name if self.use_tex else self.y.name)

        self.image = plt.imshow(self._get_z_data())
        plt.show()

    def update(self):
        """Update the rendered heat map."""
        self.image.set_data(self._get_z_data())
