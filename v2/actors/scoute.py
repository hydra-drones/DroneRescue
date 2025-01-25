from v2.states.agent_state import AgentState
from v2.agents.state_descriptor_agent import StateDescriptor
from v2.agents.decision_maker import DecisionAgent
from v2.agents.action_agent import ActionAgent
from v2.agents.communicator import CommunicationAgent
from v2.agents.strategy_agent import StrategyAgent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END, START


class Scoute:
    """Initialize Scoute agent"""

    def __init__(self, common_agent_state: AgentState, llm, verbose: bool = False):
        self.common_agent_state = common_agent_state

        self.verbose = verbose
        self.graph_is_compiled_ = False
        self.graph_ = None
        self.last_state_ = None
        self.agent_name_ = "Scoute"
        self.llm = llm

    def update_state(self, fields_to_update):
        """Set up new state"""
        for field, value in fields_to_update.items():
            if field in self.common_agent_state.keys():
                self.common_agent_state[field] = value
            else:
                print(
                    f"Field {field} not found in agent's state: {self.common_agent_state.keys}"
                )

    def compile_graph(self):
        """Compile graph"""
        descriptor_agent = StateDescriptor(self.llm, self.common_agent_state)
        decision_maker = DecisionAgent(self.llm, self.common_agent_state)
        action_agent = ActionAgent(self.llm, self.common_agent_state)
        communication_agent = CommunicationAgent(self.llm, self.common_agent_state)
        strategy_agent = StrategyAgent(self.llm, self.common_agent_state)

        graph = StateGraph(AgentState)
        graph = graph.add_node("Descriptor", descriptor_agent)
        graph = graph.add_node("DecisionMaker", decision_maker)
        graph = graph.add_node("Action", action_agent)
        graph = graph.add_node("Communicator", communication_agent)
        graph = graph.add_node("Strategist", strategy_agent)

        graph.add_edge(START, "Descriptor")
        graph = graph.add_edge("Descriptor", "DecisionMaker")
        graph = graph.add_edge("Communicator", "Action")
        graph = graph.add_edge("Strategist", "Action")
        graph.add_conditional_edges(
            "DecisionMaker",
            lambda x: x["next"],
            {
                "Action": "Action",
                "Communicator": "Communicator",
                "Strategist": "Strategist",
            },
        )

        memory = MemorySaver()
        try:
            self.graph_ = graph.compile(checkpointer=memory)
            self.graph_is_compiled_ = True
        except Exception as e:
            print(f"Graph was not compiled: {e}")

    def run_graph(self):
        """Run graph"""
        if self.graph_is_compiled_:
            config = {"configurable": {"thread_id": "1"}}
            for state in self.graph_.stream(
                self.common_agent_state, config, stream_mode="values"
            ):
                self.last_state_ = state
        else:
            print("Graph is not compiled. Please compile graph first.")

    def get_last_state(self):
        """Returns last state after the initalization"""
        return self.last_state_

    def get_data_to_take_action(self):
        action = self.last_state_.get("action_history")[-1]
        speed = self.last_state_.get("speed_history")[-1]
        observation_area = self.last_state_.get("observation_area")
        current_position = self.last_state_.get("current_position")
        return action, speed, observation_area, current_position
