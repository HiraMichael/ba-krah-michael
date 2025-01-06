import pandapower as pp
import asyncio
import mango
from grid_role import TrafoOperatorRole
from load_and_generation_role import LoadAndGenerationAgent
from vote_aggregator_role import VoteAggregatorRole
from voting_role import VotingAgent
from communication import *
import pandas as pd
import numpy as np
from config import EXPERIMENTS, LOAD_P_MW, LOAD_Q_MVAR, LOAD_SN_MVA, SGEN_P_MW
import time

np.random.seed(111)


async def main(EXP: str):
    start_time = time.time()
    # set up experiment
    max_loading_percentage = EXPERIMENTS[EXP]['max_loading_percentage']
    load_scaling_factor = EXPERIMENTS[EXP]['load_scaling_factor']
    gamma = EXPERIMENTS[EXP]['gamma']
    k_values = EXPERIMENTS[EXP]['k_values']
    canons = EXPERIMENTS[EXP]['canons']
    correlation_factor = EXPERIMENTS[EXP]['correlation_factor']

    # codec for container
    codec = mango.JSON()
    codec.add_serializer(*LoadAndGenerationDataMsg.__serializer__())
    codec.add_serializer(*AllocationMsg.__serializer__())
    codec.add_serializer(*NewStepMsg.__serializer__())
    codec.add_serializer(*CanonSpecificSharesMsg.__serializer__())
    codec.add_serializer(*YourCanonSpecificShareMsg.__serializer__())

    # import net
    print('Importing net...')
    net = pp.from_json('./data/microtopia_net.json')

    # import time-series data
    print('Importing time-series data...')
    timeseries = dict()
    timeseries[('load', 'p_mw')] = pd.read_csv(LOAD_P_MW, sep=';') * load_scaling_factor
    timeseries[('load', 'q_mvar')] = pd.read_csv(LOAD_Q_MVAR, sep=';') * load_scaling_factor
    timeseries[('load', 'sn_mva')] = pd.read_csv(LOAD_SN_MVA, sep=';') * load_scaling_factor
    timeseries[('sgen', 'p_mw')] = pd.read_csv(SGEN_P_MW, sep=';')

    #  map agent_ids onto loads and generation
    agent_ids = pd.concat([net.load['agent_id'], net.sgen['agent_id']]).dropna().unique()
    agent_indices = {}
    for agent_id in agent_ids:
        load_ind = net.load.index[net.load['agent_id'] == agent_id].to_list()
        sgen_ind = net.sgen.index[net.sgen['agent_id'] == agent_id].to_list()
        agent_indices[agent_id] = {'load': load_ind, 'sgen': sgen_ind}

    #  create new dictionary to store timeseries data per agent id
    agent_loads_and_generation = {}
    for agent_id, index_values in agent_indices.items():
        load_p_mw = {idx: timeseries[('load', 'p_mw')][str(idx)] for idx in index_values['load']}
        load_q_mvar = {idx: timeseries[('load', 'q_mvar')][str(idx)] for idx in index_values['load']}
        load_sn_mva = {idx: timeseries[('load', 'sn_mva')][str(idx)] for idx in index_values['load']}
        sgen_p_mw = {idx: timeseries[('sgen', 'p_mw')][str(idx)] for idx in index_values['sgen']}
        agent_loads_and_generation[agent_id] = {
            'loads_p_mw': load_p_mw,
            'loads_q_mvar': load_q_mvar,
            'loads_sn_mva': load_sn_mva,
            'sgens_p_mw': sgen_p_mw
        }

    # create LoadAndGenerationAgents and register to container
    load_and_generation_agents = []
    for i, (agent, loads_and_generation) in enumerate(agent_loads_and_generation.items()):
        k_i = k_values[i]
        gamma_i = gamma[i]
        load_and_gen_agent = mango.agent_composed_of(LoadAndGenerationAgent(load_and_generation_data=loads_and_generation),
                                                     VotingAgent(k_values=k_i, gamma=gamma_i))
        load_and_generation_agents.append(load_and_gen_agent)

    # create TrafoOperatorRole and register to container
    grid_agent = mango.agent_composed_of(TrafoOperatorRole(net=net,
                                                           max_loading=max_loading_percentage,
                                                           canons=canons),
                                         VoteAggregatorRole())

    async with (mango.run_with_tcp(1,
                                  grid_agent, *load_and_generation_agents,
                                  codec=codec)):
        #  pass trafo_operator_agent address to load and generation agents
        for load_and_gen_agent in load_and_generation_agents:
            load_and_gen_agent.roles[0].configure(grid_agent.addr)
        #  create agent dictionary, where integers are later used to access elements in numpy arrays
        agent_addresses = [load_and_gen_agent.addr for load_and_gen_agent in load_and_generation_agents]
        agent_dict = {agent_addresses[i]: i for i in range(len(agent_addresses))}
        #  create needs and social utility levels
        agent_needs = np.random.normal(size=len(agent_addresses))
        agent_social_utilities = np.random.normal(size=len(agent_addresses))
        #  correlate social utilities with needs
        agent_social_utilities = correlation_factor * agent_needs + np.sqrt(1 - correlation_factor**2) * agent_social_utilities
        #  scale to range between 0 and 1
        agent_needs = (agent_needs - agent_needs.min()) / (agent_needs.max() - agent_needs.min())
        agent_social_utilities = (agent_social_utilities - agent_social_utilities.min()) / (agent_social_utilities.max() - agent_social_utilities.min())

        grid_agent.roles[0].configure(agents=agent_dict,
                                      needs=agent_needs,
                                      social_utility=agent_social_utilities)

        canon_allocation_results, voting_on_weights_results, trafo_results = initialize_results(load_and_generation_agents, canons)
        rounds_of_scarcity = 0

        sim_duration = timeseries[('load', 'p_mw')].shape[0]
        print('Beginning simulation...')
        for i in range(sim_duration):
            await grid_agent.roles[0].step()
            if grid_agent.roles[0].scarcity_flag:
                rounds_of_scarcity += 1
                canon_allocation_results, voting_on_weights_results, trafo_results = update_results(
                    i, grid_agent, canon_allocation_results, voting_on_weights_results, trafo_results
                )
            print(f'\rCompleted step {i + 1}/{sim_duration} | Rounds of scarcity: {rounds_of_scarcity}', end='', flush=True)
            print("\033[F", end='', flush=True)

        end_time = time.time()
        execution_time = end_time - start_time
        with open(f'./results/{EXP}_execution_time.txt', 'w') as file:
            print(f'\nExecution Time: {execution_time:.2f} seconds\n')
            file.write(f'Execution Time: {execution_time:.2f} seconds\n')

        print('\nStoring simulation results...')
        store_results(load_and_generation_agents,
                      canons,
                      rounds_of_scarcity,
                      canon_allocation_results,
                      voting_on_weights_results,
                      trafo_results,
                      net,
                      EXP)


def initialize_results(load_and_generation_agents, canons):
    #canon_allocation_results = np.empty((len(load_and_generation_agents), 10, 1))
    canon_allocation_results = np.empty((0, 7 + len(canons)))
    #voting_on_weights_results = np.empty((len(canons), len(load_and_generation_agents) + 1, 1))
    voting_on_weights_results = np.empty((0, len(load_and_generation_agents) + 2))
    trafo_results = np.empty((0, 2))
    return canon_allocation_results, voting_on_weights_results, trafo_results


def update_results(t, grid_agent, canon_allocation_results, voting_on_weights_results, trafo_results):
    canon_allocation_time_col = np.full((len(grid_agent.roles[0].demands_at_t), 1), t + 1)
    #print(f'Time column shape: {canon_allocation_time_col.shape},'
    #      f'Demands etc. shape: {grid_agent.roles[0].demands_at_t.shape},'
    #      f'Ultimate share of demand shape: {grid_agent.roles[0].ultimate_share_of_demand_at_t.shape},'
    #      f'Canon-specific shares of demand shape: {grid_agent.roles[0].context.canon_specific_shares_of_demand.shape}')
    canon_allocation_results_at_t = np.hstack((canon_allocation_time_col,
                                               grid_agent.roles[0].demands_at_t,
                                               grid_agent.roles[0].contributions_at_t,
                                               grid_agent.roles[0].allocations_at_t,
                                               grid_agent.roles[0].ultimate_share_of_demand_at_t,
                                               grid_agent.roles[0].context.canon_specific_shares_of_demand,
                                               grid_agent.roles[0].needs.reshape(-1, 1),
                                               grid_agent.roles[0].social_utility.reshape(-1, 1)))
    canon_allocation_results = np.vstack((canon_allocation_results, canon_allocation_results_at_t))

    voting_time_col = np.full((6, ), t + 1)
    #print(f'Voting time col: {voting_time_col.shape},'
    #      f'Aggregated votes: {grid_agent.roles[1].aggregated_votes.shape},'
    #      f'Weights per agent: {grid_agent.roles[1].canon_weights_per_agent.T.shape}')
    voting_on_weights_results_at_t = np.vstack((voting_time_col,
                                                grid_agent.roles[1].aggregated_votes,
                                                grid_agent.roles[1].canon_weights_per_agent)).T
    voting_on_weights_results = np.vstack((voting_on_weights_results, voting_on_weights_results_at_t))

    trafo_results_at_t = np.hstack((grid_agent.roles[0].loading_percentage_pre_dimming,
                                    grid_agent.roles[0].loading_percentage_post_dimming))
    trafo_results = np.vstack((trafo_results, trafo_results_at_t))

    return canon_allocation_results, voting_on_weights_results, trafo_results


def store_results(load_and_generation_agents,
                  canons,
                  sim_duration,
                  canon_allocation_results,
                  voting_on_weights_results,
                  trafo_results,
                  net,
                  EXP):
    agent_ids = np.tile(np.arange(len(load_and_generation_agents)), sim_duration)
    canon_allocation_results_index = pd.MultiIndex.from_arrays(
        [canon_allocation_results[:, 0], agent_ids],
        names=["time", "agent"]
    )
    canon_allocation_results_col = ["demand", "contribution", "allocation", "share_of_demand",
                                    "sod_effort", "sod_equality", "sod_needs", "sod_social_utility",
                                    "sod_productivity", "sod_supply_and_demand", "needs", "social_utilities"]
    canon_allocation_results_df = pd.DataFrame(canon_allocation_results[:, 1:],
                                               index=canon_allocation_results_index,
                                               columns=canon_allocation_results_col)
    canon_allocation_results_df.to_csv(f'./results/{EXP}_canon_allocation_results.csv')

    canon_labels = ["effort", "equality", "needs", "social_utility", "productivity", "supply_and_demand"]
    canon_ids = np.tile(canon_labels, sim_duration)
    voting_on_weights_results_index = pd.MultiIndex.from_arrays(
        arrays=[voting_on_weights_results[:, 0], canon_ids],
        names=["time", "canon"]
    )
    voting_on_weights_results_col = ["weights"] + [f'agent{i}' for i in range(len(load_and_generation_agents))]
    voting_on_weights_results_df = pd.DataFrame(voting_on_weights_results[:, 1:],
                                                index=voting_on_weights_results_index,
                                                columns=voting_on_weights_results_col)
    voting_on_weights_results_df.to_csv(f'./results/{EXP}_voting_on_weights_results.csv')

    trafo_results_col = ["pre-dimming_perc", "post-dimming_perc"]
    trafo_results_df = pd.DataFrame(trafo_results, columns=trafo_results_col)
    trafo_results_df.to_csv(f'./results/{EXP}_trafo_results.csv')

    # storing agent data
    agent_ids = pd.concat([net.load['agent_id'], net.sgen['agent_id']]).dropna().unique()
    agent_col = ['household', 'non-household', 'charging_station', 'heatpump', 'PV']
    agent_data = pd.DataFrame(0, index=agent_ids, columns=agent_col)
    for _, row in net.load.iterrows():
        agent_data.at[row['agent_id'], row['type']] = 1
    for _, row in net.sgen.iterrows():
        # NA for static generators (although this should not occur)
        if pd.notna(row['agent_id']):
            agent_data.at[row['agent_id'], row['type']] = 1
    agent_data['gamma'] = EXPERIMENTS[EXP]['gamma']
    k_columns = [f'k_{canon_label}' for canon_label in canon_labels]
    agent_data[k_columns] = EXPERIMENTS[EXP]['k_values']
    agent_data.to_csv(f'./results/{EXP}_agent_data.csv')


asyncio.run(main('EXP2'))

