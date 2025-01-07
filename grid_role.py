from mango import AgentAddress, Role
from communication import LoadAndGenerationDataMsg, AllocationMsg, NewStepMsg, CanonSpecificSharesMsg, NewCanonWeightsMsg
import asyncio
import numpy as np
from canons import *
import pandapower as pp
from typing import Optional


class TrafoOperatorRole(Role):
    """
    The TrafoOperatorRole is responsible for ensuring that the trafo does not exceed a maximum percentage
    of a trafo's rated capacity. It collects demands and contributions of the LoadAndGenerationRole agents
    representing the loads connected to the trafo. It, then, checks whether a congestion would occur.
    In case a congestion occurs, it performs an allocation to these agents based on canons of distributive
    justice.

    Attributes of the Role need to be distinguished from those of the RoleAgent. The latter are also
    accessible to the VoteAggregatorRole.

    Attributes (of the agent to which the role is attached):
        agent_address_to_agent_index: dict
            Dictionary mapping agent addresses to indices which are used for filling the demand,
            contribution, and allocation vector.
        agent_index_to_agent_address: dict
            Inverse dictionary mapping agent indices to agent addresses.
        canon_specific_shares_of_demand: np.array
            Matrix, where the rows are agents and the columns are canons of distributive justice.
            For each round with a congestion, the matrix contains the share of demand each canon
            would assign to each agent.

    Attributes (of the role):
        net: pandapower net(attrdict)
            Net in which the trafo of the grid operator is located.
        max_sn_mva: float
            Rated capacity of the trafo.
        max_loading: float
            Maximum loading percentage of the rated capacity of the trafo.
        loads_p_mw_at_t: np.array
            Vector of active power values for the loads connected to the trafo for the current
            time step.
        loads_q_mvar_at_t: np.array
            Vector of reactive power values for the loads connected to the trafo for the current
            time step.
        loads_sn_mva_at_t: np.array
            Vector of apparent power values for the loads connected to the trafo for the current
            time step.
        sgen_p_mw_at_t: np.array
            Vector of PV generation values managed by the agents responsible for also managing
            the loads connected to the trafo for the current time step.
        demands_at_t: np.array
            Vector of demands made by the agents responsible for managing the loads/generation in
            the net. This vector is not equivalent to one of the vectors above, since one agent
            may manage multiple loads.
        contributions_at_t: np.array
            Vector of contributions made by the agents responsible for managing the loads/generation in
            the net. This vector is not equivalent to sgen_p_mw_at_t, since one agent may manage one
            or zero PV plants.
        allocations_at_t: np.array
            Vector of allocations to the agents responsible for managing the loads/generation in the
            net.
        demands_and_contributions_received: int
            Counter to track the number of demands and contributions received from agents managing
            loads/generation.
        demands_and_contributions_collected: asyncio.Event()
            Event to indicate whether all agents managing loads/generation have submitted their demands
            and contributions.
        internal_time: int
            Internal time of the role.
        canons: list[Canon]
            List of canons of distributive justice.
        new_canon_weights_received: asyncio.Event()
            Event to indicate that the VoteAggregatorRole has send a message containing the canon
            weights for the next time step.
        needs: np.array
            Vector of needs for each agent managing loads/generation, which is used for the
            canon of needs.
        social_utility: np.array
            Vector of social utilities for each agent managing loads/generation, which is used
            for the canon of social utility.
        scarcity_flag: bool
            Indicates whether or not a transformer congestion occurred (for storing simulation results only).
        ultimate_share_of_demand_at_t: np.array
            Share of demand allocated to each agent at t (for storing simulation results only).
        loading_percentage_pre_dimming: float
            Transformer loading percentage prior to fixing the congestion (for storing simulation results only).
        loading_percentage_post_dimming: float
            Transformer loading percentage after fixing the congestion (for storing simulation results only).

    Methods:
        configure(agents, needs, social_utility)
            Used to configure the TrafoOperatorRole. Agent addresses are passed to the RoleAgent,
            and the vectors of needs and social utilities are passed to the Role to be able to use
            them for computing canon-specific shares of demand.
        setup()
            Lifecycle method called when the role is added to a RoleAgent.
            Handle incoming messages.
        handle_new_canon_weights_message(content, meta)
            Upon receiving the results of the voting procedure from the VoteAggregatorRole,
            the weights of the canons of distributive justice are updated accordingly.
        handle_demand_and_contribution_msg(content, meta)
            When an agent reports its demand and contribution for a time step, the active,
            reactive, and apparent power values as well as the generation values are updated,
            and the demand and contribution of the agent are computed.
        step()
            During one simulation step, the TrafOperatorRole collects the demands and provisions
            of the agents, checks for a trafo congestion, and performs an allocation to the
            agents. In cases of scarcity, i.e., in case there is a congestion, the allocation
            is based on canons of distributive justice.
        prepare_new_step()
            Preparations are made to begin the new simulation step, e.g., clearing asyncio.Events(),
            resetting counters, etc.
        update_net()
            The net is updated based on the values reported by agents managing loads/generation.
        perform_canon_based_allocation()
            An allocation is returned, which is based on canons of distributive justice.
        compute_share_of_demand()
            Each canon is used to compute a share of demand to which an agent managing loads/generation
            has a legitimate claim. Using the canon's weights, a vector containing shares of demand
            is returned. The resulting allocation is based on this vector.
        check_sanity_of_canon_based_allocation(share_of_demand)
            Given the share of demand allocated according to the canons of distributive justice,
            the method checks whether or not the congestion would still be present.

    """

    def __init__(self,
                 net,
                 max_loading: float,
                 canons: list[Canon]
                 ):
        """
        :param net: pandapower net in which the trafo is located
        :param max_loading: maximum loading percentage of the trafo
        :param canons: canons of distributive justice
        """
        super().__init__()

        # pandapower net
        self.net = net
        # rated power of the trafo
        self.max_sn_mva = self.net.trafo.loc[0, 'sn_mva']
        # maximum loading percentage of the trafo
        self.max_loading = max_loading
        # canons of distributive justice
        self.canons = canons
        # internal time, simulation time step
        self.internal_time = 0
        # vector of needs for each agent managing load(s) and generation
        self.needs: Optional[np.array] = None
        # vector of social utilities for each agent managing load(s) and generation
        self.social_utility: Optional[np.array] = None
        # active power values for the current time step
        self.loads_p_mw_at_t: Optional[np.array] = None
        # reactive power values for the current time step
        self.loads_q_mvar_at_t: Optional[np.array] = None
        # apparent power values for the current time step
        self.loads_sn_mva_at_t: Optional[np.array] = None
        # generation values for the current time step
        self.sgen_p_mw_at_t: Optional[np.array] = None
        # agent-specific demands for the current time step
        self.demands_at_t: Optional[np.array] = None
        # agent-specific contributions for the current time step
        self.contributions_at_t: Optional[np.array] = None
        # agent-specific allocations for the current time step
        self.allocations_at_t: Optional[np.array] = None
        # dictionary storing the loads managed by each agent
        self.agents_to_loads = {}
        # counter for the number of demands and contributions received during the current time step
        self.demands_and_contributions_received = 0
        # event to indicate whether all demands and contributions have been received
        self.demands_and_contributions_collected = asyncio.Event()
        # event to indicate whether the new canon weights have been received
        self.new_canon_weights_received = asyncio.Event()

        # the following attributes are created for storing simulation results
        # flag to indicate whether a congestion occurred (only store results if this is the case)
        self.scarcity_flag = False
        # ultimate share of demand
        self.ultimate_share_of_demand_at_t: Optional[np.array] = None
        # loading percentage pre-dimming
        self.loading_percentage_pre_dimming = 0
        # loading percentage post-dimming
        self.loading_percentage_post_dimming = 0

    def configure(self,
                  agents: dict[AgentAddress, int],
                  needs: np.array,
                  social_utility: np.array
                  ):
        """
        Method to configure the TrafOperatorRole
        :param agents: dictionary mapping agent addresses to integers, which is used for storing demand,
        contribution, and allocation values at the correct index
        :param needs: agent-specific needs (used for canon of needs)
        :param social_utility: agent-specific social utilities (used for canon of social utility)
        """
        # mapping of agent addresses to agent indices
        # set as attribute of RoleAgent to be able to access in the VoteAggregatorRole
        self.context.agent_address_to_agent_index = agents
        # inverse mapping of agent indices to agent addresses (as attribute of RoleAgent)
        self.context.agent_index_to_agent_address = {index: address for address, index in agents.items()}
        # vector of agent-specific needs
        self.needs = needs
        # vector of agent-specific social utilities
        self.social_utility = social_utility
        # canon-specific shares of demand for each agent (rows) for each canon (columns)
        self.context.canon_specific_shares_of_demand = np.empty((len(agents), len(self.canons)))
        self.context.simulation_complete = asyncio.Future()
        self.ultimate_share_of_demand_at_t = np.empty((len(agents), 1))

    def setup(self):
        """
        Handle incoming messages
        """
        self.context.subscribe_message(
            self,
            self.handle_demand_and_contribution_msg,
            lambda content, meta: isinstance(content, LoadAndGenerationDataMsg),
        )
        self.context.subscribe_message(
            self,
            self.handle_new_canon_weights_msg,
            lambda content, meta: isinstance(content, NewCanonWeightsMsg),
        )

    def handle_new_canon_weights_msg(self, content, meta):
        """
        Handle incoming message from the VoteAggregatorRole which contains new canon weights
        for the next round of scarcity, and update the canon weights accordingly.
        :param content: Message content, i.e., an array containing new canon weights
        :param meta: ACL Meta information of message
        """
        # set the weight for each canon
        for canon, weight in zip(self.canons, content.new_canon_weights):
            canon.set_weight(weight)
        # set the event that canon weights have been updated
        self.new_canon_weights_received.set()

    def handle_demand_and_contribution_msg(self, content, meta):
        """
        Handle incoming message from a LoadAndGenerationRole and compute the agent's demand
        and contribution at the current time step based on the submitted load and generation
        data.
        :param content: Message content, i.e., a dictionary of dictionaries containing load
        and generation values of the agent
        :param meta: ACL Meta information of message
        :return:
        """
        # recreate agent address based on meta information and use it as dictionary key later on
        agent = AgentAddress(meta['sender_addr'], meta['sender_id'])
        # dictionary of dictionaries containing load and generation data
        demand_and_provision = content.load_and_generation_data

        # update load and generation values based on the submitted load and generation data
        # (not directly done using self.net dataframes for performance reason)
        for key, value in demand_and_provision['loads_p_mw'].items():
            self.loads_p_mw_at_t[key] = value
        for key, value in demand_and_provision['loads_q_mvar'].items():
            self.loads_q_mvar_at_t[key] = value
        for key, value in demand_and_provision['loads_sn_mva'].items():
            self.loads_sn_mva_at_t[key] = value
        for key, value in demand_and_provision['sgens_p_mw'].items():
            self.sgen_p_mw_at_t[key] = value

        # enter demand and contribution at agent index
        values_p_mw = np.array(list(demand_and_provision['loads_p_mw'].values()))
        values_q_mvar = np.array(list(demand_and_provision['loads_q_mvar'].values()))
        # compute demands as apparent power
        self.demands_at_t[self.context.agent_address_to_agent_index[agent]] = (
            np.sum(np.sqrt(values_p_mw**2 + values_q_mvar**2)))
        self.contributions_at_t[self.context.agent_address_to_agent_index[agent]] = (
            sum(demand_and_provision['sgens_p_mw'].values()))

        # increment counter for demands and contributions received
        self.demands_and_contributions_received += 1
        # set event to true once all demands and contributions have been received
        if self.demands_and_contributions_received == len(self.context.agent_address_to_agent_index):
            self.demands_and_contributions_collected.set()

        # store loads managed by agent for sanity check
        if self.internal_time == 0:
            self.agents_to_loads[self.context.agent_address_to_agent_index[agent]] = list(demand_and_provision['loads_p_mw'].keys())

    async def step(self):
        """
        Simulation step method. Within one simulation step, (1) the new simulation step is prepared,
        (2) agent-specific load and generation values are received, and associated demands and
        contributions are computed, (4) the net is updated, (5) a power flow is run to check whether
        a trafo congestion occurs given the values received, (6) an allocation is performed, which
        is based on canons of distributive justice in case of a congestion, (7) canons and canon
        weights are updated (in case of scarcity).
        :return:
        """

        # prepare the new simulation step
        self.prepare_new_step()

        # inform RoleAgents behind each LoadAndGenerationRole about the new time step,
        # who will then submit their load and generation data
        async with asyncio.TaskGroup() as tg:
            for agent in self.context.agent_address_to_agent_index.keys():
                tg.create_task(self.context.send_message(content=NewStepMsg(self.internal_time), receiver_addr=agent))
        await self.demands_and_contributions_collected.wait()

        # update net based on received values
        self.update_net()

        # run power flow and check for trafo congestion
        await asyncio.to_thread(pp.runpp, net=self.net)
        # store loading percentage pre-dimming for analysis of simulation results
        self.loading_percentage_pre_dimming = self.net.res_trafo.loc[0, 'loading_percent']
        scarcity = self.net.res_trafo.loc[0, 'loading_percent'] > self.max_loading
        # update scarcity flag for simulation results
        self.scarcity_flag = scarcity
        if scarcity:
            # in case of scarcity, perform canon-based allocation
            self.allocations_at_t = self.perform_canon_based_allocation()
        else:
            # in the absence of scarcity, allocate demand
            self.allocations_at_t = self.demands_at_t

        # inform agents about allocations
        async with asyncio.TaskGroup() as tg:
            for agent_index, agent_address in self.context.agent_index_to_agent_address.items():
                tg.create_task(self.context.send_message(content=AllocationMsg(), receiver_addr=agent_address))

        # in case of scarcity, clear the event that canon weights have been received,
        # send a message to the VoteAggregatorRole, which will trigger a new round of voting,
        # and wait for the new weights of this voting round
        if scarcity:
            self.new_canon_weights_received.clear()
            await self.context.send_message(content=CanonSpecificSharesMsg(), receiver_addr=self.context.addr)
            await self.new_canon_weights_received.wait()

        # update history-based canons using data from the current round
        for canon in self.canons:
            canon.update(scarcity=scarcity,
                         demands=self.demands_at_t,
                         contributions=self.contributions_at_t,
                         allocations=self.allocations_at_t)

        # increment internal time
        self.internal_time += 1

    def prepare_new_step(self):
        """
        Prepare a new simulation step by setting a number of arrays to zero and clearing the
        event that all demands and contributions have been received.
        """
        # numpy arrays with zeros for demands, contributions, and allocations for current time step
        self.demands_at_t = np.zeros((len(self.context.agent_address_to_agent_index), 1))
        self.contributions_at_t = np.zeros((len(self.context.agent_address_to_agent_index), 1))
        self.allocations_at_t = np.zeros((len(self.context.agent_address_to_agent_index), 1))

        # initialize np arrays for loads and generation
        # (better performance compared with directly entering values into self.net dataframes)
        self.loads_p_mw_at_t = np.zeros((len(self.net.load)))
        self.loads_q_mvar_at_t = np.zeros((len(self.net.load)))
        self.loads_sn_mva_at_t = np.zeros((len(self.net.load)))
        self.sgen_p_mw_at_t = np.zeros((len(self.net.sgen)))

        # clear the event that all demands and contributions have been received
        self.demands_and_contributions_collected.clear()
        # set the counter for demands and contributions received to 0
        self.demands_and_contributions_received = 0

    def update_net(self):
        """
        Method to update the net based on active, reactive, apparent power and generation values
        submitted by LoadAndGenerationRole.
        """
        self.net.load.loc[:, 'p_mw'] = self.loads_p_mw_at_t
        self.net.load.loc[:, 'q_mvar'] = self.loads_q_mvar_at_t
        self.net.load.loc[:, 'sn_mva'] = self.loads_sn_mva_at_t
        self.net.sgen.loc[:, 'p_mw'] = self.sgen_p_mw_at_t

    def perform_canon_based_allocation(self):
        """
        Method to perform a canon-based allocation in rounds of scarcity.
        :return: np.array containing the allocation to each RoleAgent representing a LoadAndGenerationRole.
        """
        # compute share of demand and store for simulation results
        share_of_demand = self.compute_share_of_demand()
        self.ultimate_share_of_demand_at_t = share_of_demand
        # compute allocation by conducting an element-wise multiplication of demand shares and demands
        allocation = share_of_demand * self.demands_at_t
        # perform sanity check
        self.check_sanity_of_canon_based_allocation(share_of_demand)
        return allocation

    def compute_share_of_demand(self):
        """
        Method to compute the share of demand for each agent according to each canon,
        and the ultimate share of demand based on the canon-specific shares of demand
        and the canon-specific weights.
        :return: Vector containing the ultimate share of demand for each agent.
        """
        # initialize share of demand with zero for each agent
        share_of_demand = np.zeros((len(self.context.agent_address_to_agent_index), 1))
        available_power = self.max_sn_mva * (self.max_loading / 100)

        # compute share of demand according to each canon
        for i, canon in enumerate(self.canons):
            #  compute canon-specific share of demand
            canon_specific_share_of_demand = canon.compute_share_of_demand(demands=self.demands_at_t,
                                                                           contributions=self.contributions_at_t,
                                                                           allocations=self.allocations_at_t,
                                                                           needs=self.needs,
                                                                           social_utility=self.social_utility,
                                                                           max_sn_mva=available_power)
            #  fill array of canon-specific shares of demand
            self.context.canon_specific_shares_of_demand[:, i] = canon_specific_share_of_demand.T
            #  use canon weight to compute actual share of demand according to the canon
            share_of_demand += canon.get_weight() * canon_specific_share_of_demand.reshape(-1, 1)

        # adjust the share of demand to ensure as much trafo loading as possible
        adjusted_share_of_demand = Canon.adjust_share_of_demand(share_of_demand=share_of_demand,
                                                                demand=self.demands_at_t,
                                                                max_sn_mva=available_power,
                                                                optimize=False)
        return adjusted_share_of_demand

    def check_sanity_of_canon_based_allocation(self, share_of_demand: np.array):
        """
        Use the canon-based share of demands to perform a sanity check.
        :param share_of_demand: Canon-based share of demands
        """
        sod_extended = np.zeros((len(self.loads_p_mw_at_t)))
        for agent, loads in self.agents_to_loads.items():
            for i in loads:
                sod_extended[i] = share_of_demand[agent, :]

        cos_phi_mapping = {
            'heatpump': 0.71,
            'charging_station': 1.0,
            'household': 0.95,
            'non-household': 0.95
        }

        cos_phi_vector = self.net.load['type'].map(cos_phi_mapping)
        phi_vector = np.arccos(cos_phi_vector)
        sin_vector = np.sin(phi_vector)

        self.loads_sn_mva_at_t = sod_extended * self.loads_sn_mva_at_t
        self.loads_p_mw_at_t = self.loads_sn_mva_at_t * cos_phi_vector
        self.loads_q_mvar_at_t = self.loads_sn_mva_at_t * sin_vector

        self.update_net()
        # only look at trafo
        # TODO: properly take into account PV generation
        self.net.sgen.loc[self.net.sgen['type'] == 'PV', 'in_service'] = False
        pp.runpp(self.net)

        trafo_loading_percent = self.net.res_trafo.loc[0, 'loading_percent']

        # store for simulation results
        self.loading_percentage_post_dimming = trafo_loading_percent

        # successful if loading percent is within tolerable range and below 100
        success = abs(trafo_loading_percent - self.max_loading) <= 5 and trafo_loading_percent < 100

        self.net.sgen.loc[self.net.sgen['type'] == 'PV', 'in_service'] = True
        if not success:
            print(f'\nCanon-based allocation failed: Tolerable loading percentage exceeded with loading percentage: {trafo_loading_percent}.')

