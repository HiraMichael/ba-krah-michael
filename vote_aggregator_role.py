import asyncio
from mango import Role
from communication import CanonSpecificSharesMsg, YourCanonSpecificShareMsg, NewCanonWeightsMsg, MyCanonWeightsMsg
import numpy as np
from typing import Optional


class VoteAggregatorRole(Role):
    """
    The VoteAggregatorRole aggregates the votes received from all VotingRoles to compute new canon weights
    for the next round.

    Attributes:
        canon_specific_shares_updated: asyncio.Event()
            Event to indicate whether the canon-specific shares for each agent have been updated
            for the current voting round.
        votes_received: int
            Number of votes received in the current voting round.
        aggregated_votes: np.array
            Array of votes received from each VotingRole for each canon.
        voting_complete: asyncio.Event()
            Event to indicate whether the voting is complete and new canon weights have been computed.

    Methods:
        setup()
            Lifecycle method called when the role is added to a RoleAgent.
            Handle incoming messages.
        on_start()
            Lifecycle method called once the container is started.
            Runs the VoteAggregator.
        handle_canon_specific_shares_msg(content, meta)
            Sets the event that canon-specific shares for each agent have been updated.
            Once the event is set, voting for the current round can begin.
        handle_my_canon_weights_msg(content, meta)
            Handle incoming votes for each canon when an agent submits its votes.
            Update the canon weights accordingly.
        initiate_voting()
            Send a message to each voting agent informing them about the share of demand each
            canon allocates to them in the current round.
        run()
            In each voting round, wait for canon-specific shares for each agent to be updated,
            then initiate voting, aggregate incoming votes to compute new weights,
            and inform the TrafoOperatorRole about the updated weights.

    """

    def __init__(self):
        super().__init__()

        # event to indicate whether the canon-specific shares of each agent have been updated
        self.canon_specific_shares_updated = asyncio.Event()
        # counter for the votes received in each round to determine when voting is complete,
        # i.e., when all agents have submitted their vote
        self.votes_received = 0
        # array to aggregate votes and compute the canon weights for the next rounds based on the votes
        self.aggregated_votes: Optional[np.array] = None
        # event to indicate that all votes have been received and that the canon weights for the next round have
        # been computed
        self.voting_complete = asyncio.Event()

        # TODO: add to documentation
        # the following attributes are created to store simulation results
        # canon weights per agent
        self.canon_weights_per_agent: Optional[np.array] = None

    def setup(self):
        """
        Handle incoming messages.
        """
        self.context.subscribe_message(
            self,
            self.handle_canon_specific_shares_msg,
            lambda content, meta: isinstance(content, CanonSpecificSharesMsg),
        )
        self.context.subscribe_message(
            self,
            self.handle_my_canon_weights_msg,
            lambda content, meta: isinstance(content, MyCanonWeightsMsg),
        )

    def on_start(self):
        """
        Run once the container is started.
        """
        self.context.schedule_instant_task(self.run())

    def handle_canon_specific_shares_msg(self, content, meta):
        """
        Set event once canon-specific shares have been updated to initiate voting.
        :param content: Message content
        (in this case empty, because canon-specific shares are stored in a common data container)
        :param meta: ACL Meta information of message
        """
        self.canon_specific_shares_updated.set()

    def handle_my_canon_weights_msg(self, content, meta):
        """
        Aggregate the canon weights/votes of each agent
        :param content: Message content, i.e., the agent's weights/votes for each canon
        :param meta: ACL Meta information of message
        """
        # increment votes received
        self.votes_received += 1
        # add agent's weights to each element in array self.aggregated_votes, i.e., to
        self.aggregated_votes += content.my_canon_weights.reshape(-1)
        # check if all agents have submitted their votes/weights
        if self.votes_received == len(self.context.agent_index_to_agent_address.values()):
            # normalize the canon's weights to ensure that they sum up to 1
            self.aggregated_votes = self.aggregated_votes / np.sum(self.aggregated_votes)
            # set event that voting has been complete
            self.voting_complete.set()
        # store weights per agent for analyzing simulation results
        self.canon_weights_per_agent = np.vstack((self.canon_weights_per_agent, content.my_canon_weights))

    async def initiate_voting(self):
        """
        Inform each agent about the share of demand each canon allocates to them
        """
        async with asyncio.TaskGroup() as tg:
            for agent_index, agent_address in self.context.agent_index_to_agent_address.items():
                canon_specific_shares_of_agent = self.context.canon_specific_shares_of_demand[agent_index]
                await tg.create_task(self.context.send_message(content=YourCanonSpecificShareMsg(canon_specific_shares_of_agent),
                                                             receiver_addr=agent_address))

    async def run(self):
        """
        Run the VoteAggregatorRole, i.e., wait for new voting round, inform VotingRole about new voting round,
        aggregate the votes received, and inform the TrafoOperatorRole about the new canon weights for which the agents
        have voted.
        """
        while True:
            # wait for TrafoOperatorRole to inform about updated canon-specific shares, i.e., a new voting round
            await self.canon_specific_shares_updated.wait()
            # initialize an array with a value of zero for each canon
            self.aggregated_votes = np.zeros((self.context.canon_specific_shares_of_demand.shape[1]), dtype=float)
            # initialize an array to store per agent results
            self.canon_weights_per_agent = np.empty((0, self.context.canon_specific_shares_of_demand.shape[1]), dtype=float)
            # inform each VotingRole about canon-specific shares so that they can submit their votes
            await self.initiate_voting()
            # wait until all votes have been received and new canon weights have been computed
            await self.voting_complete.wait()
            # inform TrafoOperatorRole about new canon weights
            await self.context.send_message(content=NewCanonWeightsMsg(self.aggregated_votes), receiver_addr=self.context.addr)
            # clear events and reset votes_received to 0 for next voting round
            self.canon_specific_shares_updated.clear()
            self.voting_complete.clear()
            self.votes_received = 0

