from math import log

from pgmpy.estimators import StructureScore

class AICScore(StructureScore):
    def __init__(self, data, **kwargs):
        
        super(AICScore, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):

        var_states = self.state_names[variable]
        var_cardinality = len(var_states)
        state_counts = self.state_counts(variable, parents)
        sample_size = len(self.data)
        num_parents_states = float(len(state_counts.columns))

        score = 0
        for parents_state in state_counts:  # iterate over df columns (only 1 if no parents)
            conditional_sample_size = sum(state_counts[parents_state])

            for state in var_states:
                if state_counts[parents_state][state] > 0:
                    score += state_counts[parents_state][state] * (log(state_counts[parents_state][state]) -
                                                                   log(conditional_sample_size))

        score -= num_parents_states * (var_cardinality - 1)

        return score
