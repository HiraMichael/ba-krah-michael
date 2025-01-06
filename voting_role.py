from mango import Role
from communication import YourCanonSpecificShareMsg, MyCanonWeightsMsg
import asyncio
from canons import *
from typing import Optional


class VotingAgent(Role):
    """
    The VotingRole is attached to an agent managing loads and generations. It receives the canon-specific
    shares of this agent as message. Based on these shares and its own sense of distributive justice,
    it determines the weights it would like each canon to have. The weights are then submitted as a vote
    to the VoteAggregatorRole.

    Attributes:
        k_values: np.array
            Canon-specific values determining the trade-off between self-interest and the specific
            canon of distributive justice
        discount_factor: float
            Discount factor used to compute the weight attached to current and previous voting rounds
        canon_utilities_history: np.array
            Array used to store canon utilities from previous voting rounds
        voting_rounds_counter: int
            Counter for voting rounds, which is used to compute weights for all voting rounds
        canon_weights: np.array
            Canon weights of the current voting round
        voting_complete: asyncio.Event()
            Event to indicate that the weights for each canon in the current voting round have been
            computed, after which they can be submitted to the VoteAggregatorRole

        Methods:
            setup()
                Lifecycle method called when the role is added to a RoleAgent.
                Handle incoming messages.
            on_start()
                Lifecylce method called once the container is started.
                Runs the VotingRole.
            vote(content, meta)
                Upon receiving canon-specific shares of demand of the current round, the VotingRole
                determines the weights it attaches to each canon.
            canon_utility(c, k)
                Determine the utility of a canon based on the share of demand c it allocates in the
                current round and based on the canon-specific k-value.
            run()
                In each voting round, wait for the canon-specific shares of the current round,
                determine the weights attached to each canon, and submit these weights as
                vote to the VoteAggregatorRole.

    """

    def __init__(self,
                 k_values: np.array,
                 gamma: float
                 ):
        """
        :param k_values: canon-specific values determining the trade-off between self-interest
                        and canon of distributive justice
        :param discount_factor: discount factor determining the weights attached to the current and
                        previous voting rounds
        """
        super().__init__()

        self.gamma = gamma
        self.k_values = k_values
        # array to store the weights of the current voting round
        self.canon_weights: Optional[np.array] = None
        # event which is set once voting has been completed
        self.voting_complete = asyncio.Event()

    def setup(self):
        """
        Handle incoming messages.
        """
        self.context.subscribe_message(
            self,
            self.vote,
            lambda content, meta: isinstance(content, YourCanonSpecificShareMsg)
        )

    def on_start(self):
        """
        Run once the container is started.
        :return:
        """
        self.context.schedule_instant_task(self.run())

    def vote(self, content, meta):
        """
        Upon receiving its canon-specific shares for a round, the agent updates its canon weights.
        :param content: Message content, i.e., the share of demand each canon would assign to the agent
        :param meta: ACL meta information of message
        """
        my_canon_specific_shares = content.canon_specific_shares
        # create canon_utilities array, where each value is computed using the canon utility function which
        # takes the canon-specific share and the canon-specific k value as input
        canon_utilities = np.array([[self.canon_utility(canon_specific_shares, self.k_values[i])
                                for i, canon_specific_shares in enumerate(my_canon_specific_shares)]])
        self.canon_weights = (canon_utilities + 1e-6) / (np.sum(canon_utilities) + len(canon_utilities) * 1e-6)
        # set event that voting has been complete
        self.voting_complete.set()

    def canon_utility(self, c, k):
        """
        Calculate the utility of a given canon share of demand.
        :param c: Share of demand allocated according to the canon
        :param k: Trade-off between self-interest and justice orientation for the specific canon
        :return: Utility value
        """
        u = (1 - self.gamma) * c + (self.gamma * k * c * (1 - c) + c)
        return u

    async def run(self):
        """
        Run the voting role
        """
        while True:
            # wait for voting to be completed
            await self.voting_complete.wait()
            # send message with canon weights to the VoteAggregatorRole once voting has been completed
            await self.context.send_message(content=MyCanonWeightsMsg(self.canon_weights),
                                            receiver_addr=self.context.trafo_operator_agent)
            # clear event for next voting round
            self.voting_complete.clear()
