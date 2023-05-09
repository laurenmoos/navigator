def is_feasible_trace(episode):
    return True


class Oracle:
    # specification -> abstract syntax tree -> (precondition, predicates) -> rand(next_feasible_state)
    def __init__(self, specification):
        self.specification = specification

    def _random_state(self):
        #TODO: just to get everything running, for now return a random heap state
        pass

    def _feasible_next_response(self, state):
        pass

    def next_state_given(self, state):
        return self._feasible_next_response(state)
