from enum import Enum
import re

AVAILABLE_TIME_TOKENS = {
    0: "<T+0>",
    1: "<T+1>",
    2: "<T+2>",
    3: "<T+3>",
    4: "<T+4>",
    5: "<T+5>",
    10: "<T+10>",
    20: "<T+20>",
    30: "<T+30>",
    40: "<T+40>",
    50: "<T+50>",
    51: "<LONG>",
}


class TokensMapping(Enum):
    POS_TOKEN = "<POS>"
    EGO_POS_TOKEN = "<EGO_POS>"
    TARGET_POS_TOKEN = "<TRGT>"
    ORDER_TOKEN = "<ORDER>"
    INFO_TOKEN = "<INFO>"
    SENT_MESSAGE_TOKEN = "<SND>"
    RECEIVED_MESSAGE_TOKEN = "<RCV>"
    TO_ME_TOKEN = "<TOME>"
    AGENT_STATE_TOKEN = "<AGSTATE>"
    LOCAL_STRATEGY_TOKEN = "<LOCAL_STG>"
    GLOBAL_STRATEGY_TOKEN = "<GLOBAL_STG>"
    MISSION_PROGRESS_TOKEN = "<PRGS>"
    SCOUT_TOKEN = "<SCOUT>"
    RESCUER_TOKEN = "<RESCUER>"
    COMMANDER_TOKEN = "<COMMANDER>"
    ME_FLAG_TOKEN = "<ME>"
    TO_TOKEN = "<TO>"
    MESSAGE_TOKEN = "<MESSAGE>"
    AGENT_ID_FORMAT = "AGENT#{}"

    @classmethod
    def get_all_tokens(cls) -> list[str]:
        tokens = []
        for member in cls:
            if "TOKEN" in member.name:
                tokens.append(member.value)
        tokens.extend(AVAILABLE_TIME_TOKENS.values())
        return tokens

    @classmethod
    def get_time_token(cls, time_diff: int) -> str:
        """Map time differences to predefined buckets"""
        if time_diff == 0:
            return AVAILABLE_TIME_TOKENS[0]
        elif 1 <= time_diff <= 5:
            return AVAILABLE_TIME_TOKENS[time_diff]
        elif 6 <= time_diff < 10:
            return AVAILABLE_TIME_TOKENS[10]
        elif 10 <= time_diff < 20:
            return AVAILABLE_TIME_TOKENS[20]
        elif 20 <= time_diff < 30:
            return AVAILABLE_TIME_TOKENS[30]
        elif 30 <= time_diff < 40:
            return AVAILABLE_TIME_TOKENS[40]
        elif 40 <= time_diff < 50:
            return AVAILABLE_TIME_TOKENS[50]
        else:
            return AVAILABLE_TIME_TOKENS[51]

    @classmethod
    def get_agent_token(cls, agent_id: str | int):
        return cls.AGENT_ID_FORMAT.value.format(str(agent_id))

    @classmethod
    def process_message_with_agent_tokens(cls, message: str):
        """
        Example:
        input:  Scout 10 — redeploy northward. Prioritize systematic visual sweep of northern-central and northeast sectors.
                Rescuer 2 — maintain current position. Deploy only upon confirmed contact from Scout.
        output: <AGENT#10> — redeploy northward. Prioritize systematic visual sweep of northern-central and northeast sectors.
                <AGENT#2> — maintain current position. Deploy only upon confirmed contact from Scout.'
        """
        message = message.replace("\n", " ")
        agent_token_template = cls.AGENT_ID_FORMAT.value.replace("{}", r"\2")
        message = re.sub(
            r"\b(Scout|Rescuer)\s+(\d+)\b", f"{agent_token_template}", message
        )
        return message.strip()

    @classmethod
    def process_message_with_position_tokens(cls, message: str) -> str:
        """
        Replace position coordinates with <POS> token.

        Example:
        input:  "look at the (8, 90)"
        output: "look at the <POS> 8 90"

        Args:
            message: Input message string

        Returns:
            Message with position coordinates replaced by <POS> token
        """
        pattern = r"\((\d+),\s*(\d+)\)"
        replacement = f"{cls.POS_TOKEN.value} \\1 \\2"
        processed_message = re.sub(pattern, replacement, message)
        return processed_message.strip()
