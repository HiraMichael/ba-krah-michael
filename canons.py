import numpy as np
from typing import Optional
from scipy.optimize import minimize


class Canon:
    """
    Base class for all canons.

    Attributes:
        weight: float
            Current weight of the canon.

    Methods:
        get_weight():
            Returns the current weight of the canon.
        set_weight(weight):
            Sets the weight of the canon.
        adjust_share_of_demand(share_of_demand, demand, max_sn_mva, optimize):
            Scales a share of demand by alpha to ensure as much transformer load as possible.
    """

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
        """
        Scales a share of demand by alpha to ensure as much transformer load as possible.
        :param share_of_demand: vector of demand shares
        :param demand: vector of demands
        :param max_sn_mva: rated power of the transformer
        :param optimize: bool to determine whether an optimization procedure is used for scaling share_of_demand
        :return:
        """
        epsilon = 1e-6  # avoid division by zero
        # calculate scalar alpha to scale demands
        alpha = max_sn_mva / (np.sum(share_of_demand * demand) + epsilon)
        # scale demands and cap to a maximum value of 1
        adjusted_share_of_demand = np.minimum(alpha * share_of_demand, 1)

        # optimization procedure
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
    """
    Class from which all history-based canons inherit.

    Attributes:
        discount_factor: float
            Discount factor to determine the weight attached to each previous time step.
        unweighted_share: np.array
            Matrix storing legitimate claims for each previous time step prior to weighting.
        steps: int
            Counter for time steps, which is used for computing the vector of weights.

    Methods:
        compute_share_of_demand(max_sn_mva, demands, **kwargs):
            Computes the share of demand for the current time step.
        min_max_normalize(result):
            Min-max normalization to normalize values to a range between 0 and 1.
    """

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
        """
        Computes the share of demand for the current time step.
        :param max_sn_mva: Rated power of the transformer (used for scaling).
        :param demands: Demands at t (used for scaling).
        :param kwargs: Contributions or allocations, depending on the canon.
        :return:
        """
        if self.steps == 1:
            # first step: initialize self.unweighted result and return zeros
            self.unweighted_share = np.zeros((len(demands), 1))
            return np.zeros((len(demands), 1))
        else:
            # discount factor
            step_wise_weights = self.discount_factor ** (np.arange(self.steps, 0, -1) - 1)
            share = np.matmul(step_wise_weights, self.unweighted_share.T).T
            adjusted_share = self.adjust_share_of_demand(share_of_demand=share.reshape(-1, 1),
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
    """
    Class for the canon of effort, inherits from HistoryBasedCanon.

    Methods:
        update(scarcity, demands, allocations, **kwargs):
            Computes the legitimate claim after the current round of scarcity and appends it to the history.
    """

    def __init__(self, weight: float, discount_factor: float):
        super().__init__(weight, discount_factor)

    def update(self,
               scarcity: bool,
               demands: np.array,
               allocations: np.array,
               **kwargs
               ):
        """
        Computes the legitimate claim after the current time step and appends it to the history.
        :param scarcity: bool indicating whether a congestion occurred in the current time step.
        :param demands: vector of demands
        :param allocations: vector of allocations
        """
        # only update in rounds of scarcity
        if scarcity:
            # increment steps for discount factor computation
            self.steps += 1
            # compute share for current time step and add to history
            new_unweighted_share = np.where(
                allocations > 0,
                (demands / allocations),
                (demands / 1e-6)
            )
            self.unweighted_share = np.hstack((self.unweighted_share, new_unweighted_share.reshape(-1, 1)))


class CanonOfProductivity(HistoryBasedCanon):
    """
    Class for the canon of productivity.

    Methods:
        update(contributions, allocations, **kwargs):
            Update method for the history of the canon of productivity.
    """

    def __init__(self, weight: float, discount_factor: float):
        super().__init__(weight, discount_factor)

    def update(self,
               contributions: np.array,
               allocations: np.array,
               **kwargs
               ):
        """
        Update method for the canon of productivity
        :param contributions: contributions at the current time step
        :param allocations: allocations at the current time step
        """
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
    """
    Class for the canon of supply and demand.

    Methods:
        update(scarcity, demands, contributions, **kwargs):
        Update method for the history of the canon of supply and demand.
    """

    def __init__(self, weight: float, discount_factor: float):
        super().__init__(weight, discount_factor)

    def update(self,
               scarcity: bool,
               demands: np.array,
               contributions: np.array,
               **kwargs
               ):
        """
        Update method for the canon's history
        :param scarcity: bool indicating whether a congestion occurred in the current round
        :param demands: demands at t
        :param contributions: contributions at t
        """
        # only update in rounds of scarcity
        if scarcity:
            self.steps += 1
            new_unweighted_share = np.where(
                demands > 0,
                ((contributions + 1e-6) / demands),
                ((contributions + 1e-6) / 1e-6)
            )
            self.unweighted_share = np.hstack((self.unweighted_share, new_unweighted_share.reshape(-1, 1)))


class CanonOfEquality(Canon):
    """
    Class for the canon of equality.

    Methods:
        compute_share_of_demand(max_sn_mva, demands, **kwargs):
            Compute the share of demand according to the canon of equality.
    """

    def __init__(self, weight: float):
        super().__init__(weight)

    def compute_share_of_demand(self,
                                max_sn_mva: float,
                                demands: np.array,
                                **kwargs
                                ):
        # pass a vector of ones to the adjust method, which will then be scaled by alpha
        share_of_demand = self.adjust_share_of_demand(share_of_demand=np.ones((len(demands), 1)),
                                                      demand=demands,
                                                      max_sn_mva=max_sn_mva,
                                                      optimize=False)
        return share_of_demand


class CanonOfNeeds(Canon):
    """
    Class for the canon of needs.

    Methods:
        compute_share_of_demand(needs, max_sn_mva, demands, **kwargs):
            Compute the share of demand according to the canon of needs.
    """

    def __init__(self, weight: float):
        super().__init__(weight)

    def compute_share_of_demand(self,
                                needs: np.array,
                                demands: np.array,
                                max_sn_mva: float,
                                **kwargs
                                ):
        # scale the vector of needs
        share_of_demand = self.adjust_share_of_demand(share_of_demand=needs.reshape(-1, 1),
                                                      demand=demands,
                                                      max_sn_mva=max_sn_mva,
                                                      optimize=False)
        return share_of_demand.reshape((len(share_of_demand), 1))


class CanonOfSocialUtility(Canon):
    """
    Class for the canon of social utility.

    Methods:
        compute_share_of_demand(social_utility, max_sn_mva, demands, **kwargs):
            Compute the share of demand according to the canon of social utility.
    """

    def __init__(self, weight: float):
        super().__init__(weight)

    def compute_share_of_demand(self,
                                social_utility: np.array,
                                demands: np.array,
                                max_sn_mva: float,
                                **kwargs
                                ):
        # scale the vector of social utilities
        share_of_demand = self.adjust_share_of_demand(share_of_demand=social_utility.reshape(-1, 1),
                                                      demand=demands,
                                                      max_sn_mva=max_sn_mva,
                                                      optimize=False)
        return share_of_demand.reshape((len(share_of_demand), 1))
