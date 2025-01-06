from mango import AgentAddress, Role
from communication import LoadAndGenerationDataMsg, AllocationMsg, NewStepMsg


class LoadAndGenerationAgent(Role):
    """
    The LoadAndGenerationRole manages the loads and PV plants of an agent. One agent manages one or multiple
    loads and at most one PV plant. The LoadAndGenerationRole submits the load and generation data of the
    current time step to the TrafoOperatorRole.

    Attributes:
        load_and_generation_data: dict
        Dictionary containing the time-series load and generation data for the loads and, possibly, PV plant
        managed by an agent.
    """

    def __init__(self,
                 load_and_generation_data: dict
                 ):
        """
        :param load_and_generation_data: load and generation data of the agent
        :param internal_time: internal time of the agent
        """
        super().__init__()

        # active power values of the loads managed by the agent, where the values of the dictionary
        # are numpy arrays, each of which contains the time series data for one load
        self.loads_p_mw = load_and_generation_data['loads_p_mw']
        # reactive power values for the loads managed by the agent
        self.loads_q_mvar = load_and_generation_data['loads_q_mvar']
        # apparent power values for the loads managed by the agent
        self.loads_sn_mva = load_and_generation_data['loads_sn_mva']
        # generation data for the PV plants managed by the agent
        self.sgens_p_mw = load_and_generation_data['sgens_p_mw']

    def configure(self, trafo_operator_agent: AgentAddress):
        """
        Configure the LoadAndGenerationRole by setting the TrafoOperatorRole address.
        :param trafo_operator_agent: address of the RoleAgent to which the TrafoOperatorRole is attached
        """
        self.context.trafo_operator_agent = trafo_operator_agent

    def setup(self):
        """
        Handle incoming messages.
        """
        self.context.subscribe_message(
            self,
            self.send_load_and_generation_data,
            lambda content, meta: isinstance(content, NewStepMsg),
        )
        self.context.subscribe_message(
            self,
            self.handle_allocation,
            lambda content, meta: isinstance(content, AllocationMsg)
        )

    def send_load_and_generation_data(self, content, meta):
        """
        Once the LoadAndGenerationRole is informed about a new time step, it sends the load and generation
        data of the loads and PV plants it manages for the new time step to the TrafoOperatorRole.
        :param content: Message content, i.e., new time step
        :param meta: ACL Meta information of message
        """

        # dictionary of active power values for the current time step, where the keys are the indices
        # of the associated loads and the values are the active power values associated with the loads
        loads_p_mw_at_t = {idx: self.loads_p_mw[idx][content.time] for idx in self.loads_p_mw.keys()}
        # dictionary of reactive power values for the current time step
        loads_q_mvar_at_t = {idx: self.loads_q_mvar[idx][content.time] for idx in self.loads_q_mvar.keys()}
        # dictionary of apparent power values for the current time step
        loads_sn_mva_at_t = {idx: self.loads_sn_mva[idx][content.time] for idx in self.loads_sn_mva.keys()}
        # generation values for the current time step
        sgens_p_mw_at_t = {idx: self.sgens_p_mw[idx][content.time] for idx in self.sgens_p_mw.keys()}
        # dictionary of dictionaries to combine everything
        load_and_generation_data_at_t = {'loads_p_mw': loads_p_mw_at_t,
                                         'loads_q_mvar': loads_q_mvar_at_t,
                                         'loads_sn_mva': loads_sn_mva_at_t,
                                         'sgens_p_mw': sgens_p_mw_at_t}
        # send load and generation data for the current time step to the TrafoOperatorRole
        self.context.schedule_instant_message(content=LoadAndGenerationDataMsg(load_and_generation_data_at_t),
                                              receiver_addr=self.context.trafo_operator_agent)

    def handle_allocation(self, content, meta):
        """
        Handle allocation received by the TrafoOperatorRole.
        Not implemented yet, because it is not needed for the scope of the simulation.
        :param content: Message content (empty)
        :param meta: ACL Meta information of message
        """
        pass