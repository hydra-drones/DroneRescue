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
"""
Time token mapping for temporal context in training data.

Maps time differences (in timestamp units) to standardized token representations
for consistent temporal encoding in language model training. Uses bucketing
strategy to reduce vocabulary size while preserving temporal relationships.

Token Buckets:
    - Exact tokens (0-5): Precise timing for immediate actions
    - Grouped tokens (6-50): Bucketed timing for medium delays
    - Long token (51+): Single token for extended delays
"""


class TokensMapping(Enum):
    """
    Comprehensive token mapping for drone rescue communication data.

    This enumeration defines all special tokens used to structure and standardize
    communication data from autonomous drone rescue operations. Tokens provide
    semantic markers that help language models understand the structure and
    meaning of multi-agent communication patterns.

    Token Categories:
        **Spatial Tokens**:
            - Position markers for location information
            - Target and ego-position indicators

        **Communication Tokens**:
            - Message direction indicators (sent/received)
            - Addressing and routing tokens

        **Tactical Tokens**:
            - Mission strategy markers
            - Role-based identifiers
            - Mission progress indicators

        **Message Structure Tokens**:
            - Content type identifiers (order/info)
            - Agent identification format

    Primary Use Cases:
        - **Data standardization**: Consistent format across all samples
        - **Vocabulary control**: Limited, meaningful token set for LLM training
        - **Semantic clarity**: Clear indicators of message structure and meaning
        - **Multi-agent coordination**: Explicit agent roles and relationships

    Example Usage:
        .. code-block:: python

            # Get all available tokens for vocabulary
            vocab_tokens = TokensMapping.get_all_tokens()

            # Process agent references in messages
            processed = TokensMapping.process_message_with_agent_tokens(
                "Scout 10 — redeploy northward"
            )
            # Result: "<AGENT#10> — redeploy northward"

            # Process position coordinates
            processed = TokensMapping.process_message_with_position_tokens(
                "look at the (8, 90)"
            )
            # Result: "look at the <POS> 8 90"

    Token Structure:
        All tokens follow the pattern ``<TOKEN_NAME>`` with optional parameters.
        Agent tokens use numbered format: ``<AGENT#ID>`` where ID is the agent identifier.

    .. seealso::
        AVAILABLE_TIME_TOKENS
            Time-based token mappings for temporal context
    """

    # Spatial and positional tokens
    POS_TOKEN = "<POS>"  # General position marker
    EGO_POS_TOKEN = "<EGO_POS>"  # Own agent position
    TARGET_POS_TOKEN = "<TRGT>"  # Target/objective position

    # Message content type tokens
    ORDER_TOKEN = "<ORDER>"  # Command or directive message
    INFO_TOKEN = "<INFO>"  # Informational message

    # Communication direction tokens
    SENT_MESSAGE_TOKEN = "<SND>"  # Outbound message indicator
    RECEIVED_MESSAGE_TOKEN = "<RCV>"  # Inbound message indicator
    TO_ME_TOKEN = "<TOME>"  # Message addressed to this agent

    # Agent state and strategy tokens
    AGENT_STATE_TOKEN = "<AGSTATE>"  # Agent status information
    LOCAL_STRATEGY_TOKEN = "<LOCAL_STG>"  # Individual agent strategy
    GLOBAL_STRATEGY_TOKEN = "<GLOBAL_STG>"  # Team-wide strategy
    MISSION_PROGRESS_TOKEN = "<PRGS>"  # Mission progress updates

    # Role-based agent tokens
    SCOUT_TOKEN = "<SCOUT>"  # Scout role identifier
    RESCUER_TOKEN = "<RESCUER>"  # Rescuer role identifier
    COMMANDER_TOKEN = "<COMMANDER>"  # Commander role identifier

    # Message structure tokens
    ME_FLAG_TOKEN = "<ME>"  # Self-reference marker
    TO_TOKEN = "<TO>"  # Message destination marker
    MESSAGE_TOKEN = "<MESSAGE>"  # Message content delimiter

    # Agent identification format template
    AGENT_ID_FORMAT = "AGENT#{}"  # Template for agent ID tokens

    @classmethod
    def get_all_tokens(cls) -> list[str]:
        """
        Retrieve complete vocabulary of all available tokens.

        Combines semantic tokens from this enumeration with temporal tokens
        to create a comprehensive vocabulary list for language model training
        and tokenization processes.

        :returns: Complete list of all token strings
        :rtype: list[str]

        Token Categories Included:
            - All semantic tokens from TokensMapping enum
            - All temporal tokens from AVAILABLE_TIME_TOKENS
            - Agent ID template format

        Example:
            .. code-block:: python

                tokens = TokensMapping.get_all_tokens()
                print(f"Total vocabulary size: {len(tokens)}")
                print("Sample tokens:", tokens[:5])

                # Output:
                # Total vocabulary size: 23
                # Sample tokens: ['<POS>', '<EGO_POS>', '<TRGT>', '<ORDER>', '<INFO>']

        .. note::
            This method filters enum members to include only those with "TOKEN"
            in their name, ensuring consistent token identification.
        """
        tokens = []
        for member in cls:
            if "TOKEN" in member.name:
                tokens.append(member.value)
        tokens.extend(AVAILABLE_TIME_TOKENS.values())
        return tokens

    @classmethod
    def get_time_token(cls, time_diff: int) -> str:
        """
        Map time differences to standardized temporal tokens.

        Converts raw time differences into bucketed token representations that
        provide temporal context while maintaining manageable vocabulary size.
        Uses progressive bucketing strategy with finer granularity for smaller
        time differences.

        :param time_diff: Time difference in timestamp units
        :type time_diff: int

        :returns: Corresponding time token string
        :rtype: str

        Bucketing Strategy:
            - **0 seconds**: ``<T+0>`` (simultaneous)
            - **1-5 seconds**: ``<T+1>`` to ``<T+5>`` (immediate)
            - **6-9 seconds**: ``<T+10>`` (short delay)
            - **10-19 seconds**: ``<T+20>`` (medium delay)
            - **20-29 seconds**: ``<T+30>`` (longer delay)
            - **30-39 seconds**: ``<T+40>`` (extended delay)
            - **40-49 seconds**: ``<T+50>`` (long delay)
            - **50+ seconds**: ``<LONG>`` (very long delay)

        Example:
            .. code-block:: python

                # Immediate timing
                token = TokensMapping.get_time_token(0)   # "<T+0>"
                token = TokensMapping.get_time_token(3)   # "<T+3>"

                # Bucketed timing
                token = TokensMapping.get_time_token(8)   # "<T+10>"
                token = TokensMapping.get_time_token(15)  # "<T+20>"
                token = TokensMapping.get_time_token(100) # "<LONG>"

        Design Rationale:
            The bucketing strategy balances temporal precision with vocabulary
            efficiency. Fine granularity for immediate actions (0-5s) captures
            rapid interaction patterns, while progressive bucketing for longer
            delays reduces token complexity without losing temporal awareness.
        """
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
        """
        Generate standardized agent identification token.

        Creates consistent agent ID tokens using the predefined format template.
        Accepts both string and integer agent identifiers for flexibility.

        :param agent_id: Agent identifier (numeric or string)
        :type agent_id: str | int

        :returns: Formatted agent token
        :rtype: str

        Token Format:
            ``<AGENT#{ID}>`` where {ID} is the provided agent identifier.

        Example:
            .. code-block:: python

                # Numeric agent ID
                token = TokensMapping.get_agent_token(5)     # "<AGENT#5>"

                # String agent ID
                token = TokensMapping.get_agent_token("10")  # "<AGENT#10>"

                # Use in message processing
                agent_token = TokensMapping.get_agent_token(agent_id)
                message = f"{agent_token} reporting status"

        .. note::
            Agent IDs are converted to strings internally to ensure consistent
            formatting regardless of input type.
        """
        return cls.AGENT_ID_FORMAT.value.format(str(agent_id))

    @classmethod
    def process_message_with_agent_tokens(cls, message: str):
        """
        Replace role-based agent references with standardized agent tokens.

        Transforms natural language agent references (e.g., "Scout 10", "Rescuer 2")
        into consistent token format using regex pattern matching. This standardization
        enables consistent agent identification across all training data.

        :param message: Input message containing role-based agent references
        :type message: str

        :returns: Message with agent references replaced by tokens
        :rtype: str

        Pattern Matching:
            Recognizes patterns: ``(Scout|Rescuer) <number>``
            Replaces with: ``<AGENT#<number>>``

        Example:
            .. code-block:: python

                # Single agent reference
                message = "Scout 10 — redeploy northward"
                processed = TokensMapping.process_message_with_agent_tokens(message)
                # Result: "<AGENT#10> — redeploy northward"

                # Multiple agent references
                message = "Scout 10 report to Rescuer 2 immediately"
                processed = TokensMapping.process_message_with_agent_tokens(message)
                # Result: "<AGENT#10> report to <AGENT#2> immediately"

                # Complex message
                message = '''Scout 10 — redeploy northward. Prioritize systematic visual sweep.
                Rescuer 2 — maintain current position. Deploy only upon confirmed contact.'''
                processed = TokensMapping.process_message_with_agent_tokens(message)
                # Result: "<AGENT#10> — redeploy northward... <AGENT#2> — maintain current position..."

        Processing Features:
            - **Multi-line handling**: Converts newlines to spaces for consistent format
            - **Role flexibility**: Handles both Scout and Rescuer role prefixes
            - **Number extraction**: Preserves agent numbers in token format
            - **Whitespace cleanup**: Strips leading/trailing whitespace

        .. note::
            This method preserves all other message content while only transforming
            agent references, maintaining message semantics and structure.
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
        Replace coordinate pairs with standardized position tokens.

        Transforms parenthetical coordinate notation into structured position
        tokens that maintain spatial information in a format optimized for
        language model processing.

        :param message: Input message containing coordinate references
        :type message: str

        :returns: Message with coordinates replaced by position tokens
        :rtype: str

        Pattern Matching:
            Recognizes: ``(x, y)`` coordinate format
            Replaces with: ``<POS> x y`` space-separated format

        Example:
            .. code-block:: python

                # Single coordinate reference
                message = "look at the (8, 90)"
                processed = TokensMapping.process_message_with_position_tokens(message)
                # Result: "look at the <POS> 8 90"

                # Multiple coordinates
                message = "move from (10, 20) to (50, 60)"
                processed = TokensMapping.process_message_with_position_tokens(message)
                # Result: "move from <POS> 10 20 to <POS> 50 60"

                # Complex spatial reference
                message = "target spotted at (15, 75), establishing perimeter at (20, 80)"
                processed = TokensMapping.process_message_with_position_tokens(message)
                # Result: "target spotted at <POS> 15 75, establishing perimeter at <POS> 20 80"

        Token Benefits:
            - **Vocabulary reduction**: Single position token vs. varied coordinate formats
            - **Consistent parsing**: Predictable token structure for coordinate extraction
            - **Spatial awareness**: Preserves coordinate values for spatial reasoning
            - **Flexible spacing**: Handles varying whitespace in coordinate notation

        .. note::
            This transformation maintains coordinate precision while standardizing
            the representation format for improved model training consistency.
        """
        pattern = r"\((\d+),\s*(\d+)\)"
        replacement = f"{cls.POS_TOKEN.value} \\1 \\2"
        processed_message = re.sub(pattern, replacement, message)
        return processed_message.strip()
