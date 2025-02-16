NUMBER_OF_AGENTS = (2, 10)
AGENT_POSITIONS = ((1, 1), (100, 100))  # range
NUMBER_OF_TARGETS = (1, 10)
GLOBAL_MISSION = {
    "scout": "scout postitions of each target",
    "rescuer": "scan all targets using coordinates provided by scout",
}
ROLES = ["scout", "rescuer"]
SIZE_OF_MISSION_AREA = (100, 100)

AGENT_SYMBOLS = {"scout": "o", "rescuer": "*", "target": "x"}

AGENT_COLORS = {"scout": "blue", "rescuer": "green", "target": "red"}

__all__ = [
    "NUMBER_OF_AGENTS",
    "AGENT_POSITIONS",
    "NUMBER_OF_TARGETS",
    "GLOBAL_MISSION",
    "ROLES",
    "SIZE_OF_MISSION_AREA",
    "AGENT_SYMBOLS",
    "AGENT_COLORS",
]
