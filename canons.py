import numpy as np
from typing import Optional
from scipy.optimize import minimize


class Canon:

    def __init__(self, weight: float):
        self.weight = weight

    def get_weight(self):
        return self.weight

    def set_weight(self, new_weight: float):
        self.weight = new_weight

    def update(self, **kwargs):
        pass

    def compute_share_of_demand(self, **kwargs):
        pass

    @staticmethod
    def adjust_share_of_demand(share_of_demand: np.array,
                               demand: np.array,
                               max_sn_mva: float,
                               optimize: bool
                               ):
        epsilon = 1e-6  # avoid division by zero
        alpha = max_sn_mva / (np.sum(share_of_demand * demand) + epsilon)
        adjusted_share_of_demand = np.minimum(alpha * share_of_demand, 1)

        if optimize:
            share_of_demand = share_of_demand.flatten()
            demand = demand.flatten()

            def objective(a):
                return (np.sum(a * share_of_demand * demand) - max_sn_mva) ** 2

            constraints = [{'type': 'ineq', 'fun': lambda a: 1 - a * share_of_demand}]
            bounds = [(0, None) for _ in range(len(share_of_demand))]
            a0 = np.ones(len(share_of_demand))
            result = minimize(objective, a0, bounds=bounds, constraints=constraints)
            if result.success:
                optimized_share_of_demand = result.x * share_of_demand
                return optimized_share_of_demand.reshape(-1, 1)

        return adjusted_share_of_demand


class HistoryBasedCanon(Canon):

    def __init__(self, weight: float, discount_factor: float):
        super().__init__(weight)

        self.discount_factor = discount_factor
        self.unweighted_share: Optional[np.array] = None
        self.steps = 1

    def compute_share_of_demand(self,
                                max_sn_mva: float,
                                demands: np.array,
                                **kwargs
                                ):
        if self.steps == 1:
            # first step: initialize self.unweighted result and return zeros
            self.unweighted_share = np.zeros((len(demands), 1))
            return np.zeros((len(demands), 1))
        else:
            # discount factor version
            step_wise_weights = self.discount_factor ** (np.arange(self.steps, 0, -1) - 1)
            share = np.matmul(step_wise_weights, self.unweighted_share.T)
            #normalized_share = self.min_max_normalize(share.T)
            normalized_share = share.T
            adjusted_share = self.adjust_share_of_demand(share_of_demand=normalized_share.reshape(-1, 1),
                                                         demand=demands,
                                                         max_sn_mva=max_sn_mva,
                                                         optimize=False)
            return adjusted_share

    @staticmethod
    def min_max_normalize(result: np.array):
        min_value = np.min(result)
        max_value = np.max(result)
        result = np.where(
            min_value != max_value,
            ((result - min_value) / (max_value - min_value)),
            result
        )
        return result


class CanonOfEffort(HistoryBasedCanon):

    def __init__(self, weight: float, discount_factor: float):
        super().__init__(weight, discount_factor)

    def update(self,
               scarcity: bool,
               demands: np.array,
               allocations: np.array,
               **kwargs
               ):
        if scarcity:
            self.steps += 1
            new_unweighted_share = np.where(
                allocations > 0,
                (demands / allocations),
                (demands / 1e-6)
            )
            self.unweighted_share = np.hstack((self.unweighted_share, new_unweighted_share.reshape(-1, 1)))


class CanonOfProductivity(HistoryBasedCanon):

    def __init__(self, weight: float, discount_factor: float):
        super().__init__(weight, discount_factor)

    def update(self,
               contributions: np.array,
               allocations: np.array,
               **kwargs
               ):
        # initialize unweighted share for first update round, which is necessary for the canon
        # as, unlike the other history-based canons, it is updated in all rounds
        if self.steps == 1:
            self.unweighted_share = np.zeros((len(allocations), 1))
        self.steps += 1
        new_unweighted_share = np.where(
            allocations > 0,
            (contributions / (allocations + 1e-6)),  # somehow still needed to avoid divide by zero
            (contributions / 1e-6)
        )
        self.unweighted_share = np.hstack((self.unweighted_share, new_unweighted_share.reshape(-1, 1)))


class CanonOfSupplyAndDemand(HistoryBasedCanon):

    def __init__(self, weight: float, discount_factor: float):
        super().__init__(weight, discount_factor)

    def update(self,
               scarcity: bool,
               demands: np.array,
               contributions: np.array,
               **kwargs
               ):
        if scarcity:
            self.steps += 1
            new_unweighted_share = np.where(
                demands > 0,
                ((contributions + 1e-6) / demands),
                ((contributions + 1e-6) / 1e-6)
            )
            self.unweighted_share = np.hstack((self.unweighted_share, new_unweighted_share.reshape(-1, 1)))


class CanonOfEquality(Canon):

    def __init__(self, weight: float):
        super().__init__(weight)

    def compute_share_of_demand(self,
                                max_sn_mva: float,
                                demands: np.array,
                                **kwargs
                                ):
        share_of_demand = self.adjust_share_of_demand(share_of_demand=np.ones((len(demands), 1)),
                                                      demand=demands,
                                                      max_sn_mva=max_sn_mva,
                                                      optimize=False)
        return share_of_demand


class CanonOfNeeds(Canon):

    def __init__(self, weight: float):
        super().__init__(weight)

    def compute_share_of_demand(self,
                                needs: np.array,
                                demands: np.array,
                                max_sn_mva: float,
                                **kwargs
                                ):
        share_of_demand = self.adjust_share_of_demand(share_of_demand=needs.reshape(-1, 1),
                                                      demand=demands,
                                                      max_sn_mva=max_sn_mva,
                                                      optimize=False)
        return share_of_demand.reshape((len(share_of_demand), 1))


class CanonOfSocialUtility(Canon):

    def __init__(self, weight: float):
        super().__init__(weight)

    def compute_share_of_demand(self,
                                social_utility: np.array,
                                demands: np.array,
                                max_sn_mva: float,
                                **kwargs
                                ):
        share_of_demand = self.adjust_share_of_demand(share_of_demand=social_utility.reshape(-1, 1),
                                                      demand=demands,
                                                      max_sn_mva=max_sn_mva,
                                                      optimize=False)
        return share_of_demand.reshape((len(share_of_demand), 1))
