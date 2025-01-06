import pandapower as pp
import pandas as pd
import numpy as np
from config import LOAD_P_MW, LOAD_Q_MVAR, LOAD_SN_MVA, SGEN_P_MW, MAX_LOADING_PERC, LOAD_SCALING_FACTOR
import time


def update_net(net, timeseries, time):
    # update load data
    load_p_mw = timeseries[('load', 'p_mw')].loc[time, :].to_numpy()
    net.load.loc[:, 'p_mw'] = load_p_mw
    load_q_mvar = timeseries[('load', 'q_mvar')].loc[time, :].to_numpy()
    net.load.loc[:, 'q_mvar'] = load_q_mvar
    load_sn_mva = timeseries[('load', 'sn_mva')].loc[time, :].to_numpy()
    net.load.loc[:, 'sn_mva'] = load_sn_mva
    # update sgen data
    sgen_p_mw = timeseries[('sgen', 'p_mw')].loc[time, :].to_numpy()
    net.sgen.loc[:, 'p_mw'] = sgen_p_mw

def store_results(allocation_results, time, demands, contributions, allocations):
    demands = np.array(demands)
    contributions = np.array(contributions)
    allocations = np.array(allocations)

    for agent, (d, c, a) in enumerate(zip(demands, contributions, allocations)):
        allocation_results.append({
            'time': time,
            'agent': agent,
            'demand': d,
            'contribution': c,
            'allocation': a,
            'share_of_demand': a / d
        })



if __name__ == '__main__':
    start_time = time.time()

    # import net
    print('Importing net...')
    net = pp.from_json('./data/microtopia_net.json')

    cos_phi_mapping = {
        'heatpump': 0.71,
        'charging_station': 1.0,
        'household': 0.95,
        'non-household': 0.95
    }

    cos_phi_vector = net.load['type'].map(cos_phi_mapping)
    phi_vector = np.arccos(cos_phi_vector)
    sin_vector = np.sin(phi_vector)

    # import time-series data
    print('Importing time-series data...')
    timeseries = dict()
    timeseries[('load', 'p_mw')] = pd.read_csv(LOAD_P_MW, sep=';') * LOAD_SCALING_FACTOR
    timeseries[('load', 'q_mvar')] = pd.read_csv(LOAD_Q_MVAR, sep=';') * LOAD_SCALING_FACTOR
    timeseries[('load', 'sn_mva')] = pd.read_csv(LOAD_SN_MVA, sep=';') * LOAD_SCALING_FACTOR
    timeseries[('sgen', 'p_mw')] = pd.read_csv(SGEN_P_MW, sep=';')

    agent_ids = pd.concat([net.load['agent_id'], net.sgen['agent_id']]).dropna().unique()
    allocation_results = []

    max_sn_mva = net.trafo.loc[0].sn_mva
    min_load_mva = 0.0e-3
    congestion_times = []
    pre_dimming_percentages = []
    post_dimming_percentages = []
    sim_duration = timeseries[('load', 'p_mw')].shape[0]
    rounds_of_scarcity = 0
    for i in range(sim_duration):
        update_net(net, timeseries, i)
        pp.runpp(net)
        trafo_load = net.res_trafo.loc[0, 'loading_percent']
        if trafo_load > MAX_LOADING_PERC:
            rounds_of_scarcity += 1

            demands = []
            contributions = []
            allocations = []

            # store demands and contributions for each agent
            for agent in agent_ids:
                agent_demand = net.load.loc[net.load['agent_id'] == agent, 'sn_mva'].sum()
                demands.append(agent_demand)
                agent_contribution = net.sgen.loc[net.sgen['agent_id'] == agent, 'p_mw'].sum()
                contributions.append(agent_contribution)


            congestion_times.append(i + 1)
            pre_dimming_percentages.append(trafo_load)
            overload = ((trafo_load - MAX_LOADING_PERC) / 100)
            required_curtailment_mva = overload * max_sn_mva

            curtailment_list = []
            sorted_loads = net.load.sort_values(by=['p_mw'], ascending=False)
            for load_index, row in sorted_loads.iterrows():
                if required_curtailment_mva > 0:
                    assert row.p_mw >= 0
                    assert row.q_mvar >= 0
                    curtailable_mva = np.sqrt(row.p_mw ** 2 + row.q_mvar ** 2)
                    if curtailable_mva > min_load_mva:
                        curtailment_list.append(load_index)
                        curtailed_mva = curtailable_mva - min_load_mva
                        required_curtailment_mva -= curtailed_mva
                else:
                    break

            for load_index in curtailment_list:
                net.load.loc[load_index, 'sn_mva'] = min_load_mva
                net.load.loc[load_index, 'p_mw'] = min_load_mva * cos_phi_vector[load_index]
                net.load.loc[load_index, 'q_mvar'] = min_load_mva * sin_vector[load_index]

            # re-run power flow
            net.sgen.loc[net.sgen['type'] == 'PV', 'in_service'] = False
            pp.runpp(net)
            post_dimming_perc = net.res_trafo.loc[0, 'loading_percent']
            post_dimming_percentages.append(post_dimming_perc)
            net.sgen.loc[net.sgen['type'] == 'PV', 'in_service'] = True

            # map onto agents
            for agent in agent_ids:
                agent_allocation = net.load.loc[net.load['agent_id'] == agent, 'sn_mva'].sum()
                allocations.append(agent_allocation)

            # store results
            store_results(allocation_results, i + 1, demands, contributions, allocations)
        print(f'\rCompleted step {i + 1}/{sim_duration} | Rounds of scarcity: {rounds_of_scarcity}', end='',
              flush=True)

    end_time = time.time()
    execution_time = end_time - start_time
    with open('./results/BL_execution_time.txt', 'w') as file:
        print(f'\nExecution Time: {execution_time:.2f} seconds\n')
        file.write(f'Execution Time: {execution_time:.2f} seconds\n')

    allocation_results = pd.DataFrame(allocation_results)
    allocation_results.to_csv('./results/BL_allocation_results.csv')

    congestion_times = np.array(congestion_times)
    pre_dimming_percentages = np.array(pre_dimming_percentages)
    post_dimming_percentages = np.array(post_dimming_percentages)

    trafo_results = pd.DataFrame({
        'time': congestion_times,
        'pre-dimming_perc': pre_dimming_percentages,
        'post-dimming_perc': post_dimming_percentages
    })
    trafo_results.to_csv('./results/BL_trafo_results.csv')

