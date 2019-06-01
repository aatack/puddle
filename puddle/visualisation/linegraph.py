from puddle.construction.space import Space, Scalar
from puddle.construction.variable import Variable
from matplotlib import rc
import numpy as np
import matplotlib.pyplot as plt


class LineGraph:
    def __init__(
        self,
        x,
        y,
        x_index=[],
        y_index=[],
        x_range=None,
        y_range=None,
        use_tex=False,
        segments=100,
        colour="blue",
    ):
        """Create a plot of one variable against another."""
        if not isinstance(x, Variable) or not isinstance(y, Variable):
            raise ValueError("both x and y values must be Variables")
        if not isinstance(x, Scalar):
            raise ValueError("x value must be a Space in order to be plotted")
        if isinstance(y, Space) or len(y.shape) > 0:
            raise ValueError("y value must be a scalar and dependent")

        self.x = x
        self.x_index = x_index
        self.x_range = x_range
        self.xs = np.linspace(self.x.lower, self.x.upper, segments)

        self.y = y
        self.y_index = y_index
        self.y_range = y_range
        self.y.export()

        self.use_tex = use_tex

        self.segments = segments
        self.colour = colour

        self.figure = None
        self.line = None

        if self.use_tex:
            rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
            rc("text", usetex=True)

    def _get_y_data(self):
        """Get the data to plot on the y-axis."""
        return self.y(self.xs)

    def open(self):
        """Plot the y-value as a function of the x-value."""
        plt.ion()
        self.figure = plt.figure()
        axes = self.figure.add_subplot(111)
        self.line, = axes.plot(self.xs, self._get_y_data(), self.colour)

        if self.y_range is not None:
            plt.ylim(*self.y_range)
        plt.xlim(self.x.lower, self.x.upper)

        plt.xlabel(self.x.tex_name if self.use_tex else self.x.name)
        plt.ylabel(self.y.tex_name if self.use_tex else self.y.name)

        self.figure.canvas.draw()

    def update(self):
        """Update the data shown on the graph."""
        self.line.set_ydata(self._get_y_data())
        self.figure.canvas.draw()
