from puddle.construction.space import Space, Scalar
from puddle.construction.variable import Variable
from matplotlib import rc
import numpy as np
import matplotlib.pyplot as plt


class Plot:
    def __init__(
        self, x, y, x_index=[], y_index=[], x_range=None, y_range=None, use_tex=False
    ):
        """Create a plot of one variable against another."""
        self.x = x
        self.x_index = x_index
        self.x_range = x_range
        self.y = y
        self.y_index = y_index
        self.y_range = y_range

        self.use_tex = use_tex

        if not isinstance(self.x, Variable) or not isinstance(self.y, Variable):
            raise ValueError("both x and y values must be Variables")

    def line(self, segments=100, colour="blue"):
        """Plot the y-value as a function of the x-value."""
        if not isinstance(self.x, Scalar):
            raise ValueError("x value must be a Space in order to be plotted")
        if isinstance(self.y, Space) or len(self.y.shape) > 0:
            raise ValueError("y value must be a scalar and dependent")

        self.y.export()
        xs = np.linspace(self.x.lower, self.x.upper, segments)
        ys = self.y(xs)

        if self.use_tex:
            rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
            rc("text", usetex=True)

        plt.plot(xs, ys)

        if self.y_range is not None:
            plt.ylim(*self.y_range)
        plt.xlim(self.x.lower, self.x.upper)

        plt.xlabel(self.x.tex_name if self.use_tex else self.x.name)
        plt.ylabel(self.y.tex_name if self.use_tex else self.y.name)

        plt.show()
