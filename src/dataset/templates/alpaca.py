from pydantic import BaseModel


class AlpacaTemplate(BaseModel):
    """
    Alpaca dataset format template for instruction-following training data.

    This template defines the standard Alpaca format structure used for training
    large language models on instruction-following tasks. The format consists of
    four components that together create a complete training example for
    supervised fine-tuning.

    The Alpaca format is designed to train models to follow instructions by
    providing context (input), guidance (instruction), and expected behavior
    (output) within a structured framework (system).

    Attributes:
        instruction (str): Task description or directive for the model
        input (str): Contextual information or data for the task
        output (str): Expected model response or completion
        system (str): System-level context or role definition

    Format Structure:
        - **instruction**: Describes what the model should do
        - **input**: Provides the contextual data or scenario
        - **output**: Shows the expected model behavior/response
        - **system**: Defines system role or constraints (often empty)

    Example:
        .. code-block:: python

            # Drone rescue communication training sample
            sample = AlpacaTemplate(
                instruction="",  # Currently unused in this dataset
                input="<T+0> <POS> 10 20 <T+5> <RCV> <AGENT#2> Status report",
                output="<SND> <TO> <AGENT#2> All clear, proceeding to target",
                system=""  # Currently unused in this dataset
            )

            # Serialize for training
            training_data = sample.model_dump()

    Use Cases:
        - **Conversational AI**: Teaching models to respond appropriately
        - **Task Completion**: Training models to follow multi-step instructions
        - **Domain Adaptation**: Fine-tuning models for specific domains
        - **Behavior Alignment**: Teaching desired response patterns

    .. note::
        In the current drone rescue dataset, the ``instruction`` and ``system``
        fields are typically empty strings, with the primary learning signal
        coming from the ``input`` (contextual events) and ``output`` (target response).

    .. seealso::
        Stanford Alpaca Dataset: https://github.com/tatsu-lab/stanford_alpaca
            Original Alpaca format specification and examples
    """

    instruction: str
    input: str
    output: str
    system: str
