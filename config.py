import numpy as np
import random
from canons import *

# import data paths
LOAD_P_MW = './data/microtopia_load_p_mw.csv'
LOAD_Q_MVAR = './data/microtopia_load_q_mvar.csv'
LOAD_SN_MVA = './data/microtopia_load_sn_mva.csv'
SGEN_P_MW = './data/microtopia_sgen_p_mw.csv'

# assumptions on active and reactive power consumption
COS_PHI = 0.95
PHI = np.arccos(COS_PHI)
SIN_PHI = np.sin(PHI)

# load scaling factor
LOAD_SCALING_FACTOR = 1

# max loading percentage of trafo
MAX_LOADING_PERC = 85

# discount factor used for both canons and voting
DISCOUNT_FACTOR = 0.5

# correlation between needs and social utilities
CORRELATION_FACTOR = 0.25

# create canon objects with initial weights and function
NO_CANONS = 6
initial_weight = 1 / NO_CANONS
canon_of_effort = CanonOfEffort(initial_weight, DISCOUNT_FACTOR)
canon_of_equality = CanonOfEquality(initial_weight)
canon_of_needs = CanonOfNeeds(initial_weight)
canon_of_social_utility = CanonOfSocialUtility(initial_weight)
canon_of_productivity = CanonOfProductivity(initial_weight, DISCOUNT_FACTOR)
canon_of_supply_and_demand = CanonOfSupplyAndDemand(initial_weight, DISCOUNT_FACTOR)
canons = [canon_of_effort, canon_of_equality, canon_of_needs, canon_of_social_utility,
          canon_of_productivity, canon_of_supply_and_demand]

# number of agents
NO_AGENTS = 111

# experiments
# EXP1: only self-interested agents (k_values irrelevant)
# EXP2: around half self-interested, around half only justice-oriented
# generate k_values for EXP2, where the justice-oriented agents are community-oriented,
# i.e., canons of equality, needs, social utility have higher k value than other canons
# these canons have indices 1, 2, 3
# further, we restrict k to a range between 1 and 10
k_values = np.zeros((NO_AGENTS, NO_CANONS))
for i in range(NO_AGENTS):
    for j in range(NO_CANONS):
        if j in [1, 2, 3]:
            k_values[i, j] = np.random.randint(6, 11)
        else:
            k_values[i, j] = np.random.randint(1, 6)
gamma = np.array([0] * (NO_AGENTS // 2) + [1] * ((NO_AGENTS // 2) + 1))
random.shuffle(gamma)
EXPERIMENTS = {
    'EXP1': {'max_loading_percentage': MAX_LOADING_PERC,
             'load_scaling_factor': LOAD_SCALING_FACTOR,
             'gamma': np.zeros(NO_AGENTS),
             'k_values': np.ones((NO_AGENTS, NO_CANONS)),
             'canons': canons,
             'correlation_factor': CORRELATION_FACTOR
             },
    'EXP2': {'max_loading_percentage': MAX_LOADING_PERC,
             'load_scaling_factor': LOAD_SCALING_FACTOR,
             'gamma': gamma,
             'k_values': k_values,
             'canons': canons,
             'correlation_factor': CORRELATION_FACTOR
             }
}
