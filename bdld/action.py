"""Module containing just the abstract base class for all actions"""


class Action:
    """Abstract base class for all actions"""
    def run(self, step: int):
        """Needs to be defined for all actions"""
        raise NotImplementedError()

    def final_run(self, step: int):
        """If not implemented, do nothing"""
        pass
