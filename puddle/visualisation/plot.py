class Plot:
    def scatter(self, inputs):
        """
        Plot the given points as a scatter plot, calculating missing values.

        Inputs are given as a dictionary in which the keys are variables and the
        values are lists of input values.
        """
        raise NotImplementedError()

    def line(self, inputs):
        """
        Plot the given points as a line, calculating missing values as needed.

        Inputs are given as a dictionary in which the keys are variables and the
        values are lists of input values.
        """
        raise NotImplementedError()

    def heat_map(self, inputs):
        """
        Plot the given points as a heat map, calculating missing values as needed.

        Inputs are given as a dictionary in which the keys are variables and the
        values are lists of input values.
        """
        raise NotImplementedError()
