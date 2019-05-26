class PuddleRepository:

    variables = set()
    independent_variables = set()
    equations = set()

    most_recent_system = None

    @staticmethod
    def register_variable(variable, is_independent=False, is_equation=False):
        """Register a variable that has been created."""
        PuddleRepository.variables.add(variable)
        if is_independent:
            PuddleRepository.independent_variables.add(variable)
        if is_equation:
            PuddleRepository.equations.add(variable)

    @staticmethod
    def clear_cache():
        PuddleRepository.variables = set()
        PuddleRepository.independent_variables = set()
        PuddleRepository.equations = set()
