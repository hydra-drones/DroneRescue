from bisect import bisect_left

from src.dataset.base.splitters import BaseSpliter
from src.dataset.base.models import SplittedData, TimelineData, PostProcessedSample


class SlicingWindowSplitter(
    BaseSpliter[SplittedData, list[TimelineData], list[PostProcessedSample]]
):
    """
    Sliding window splitter for creating temporal training sequences.

    This splitter processes timeline data by creating training windows around target
    events. For each target event (e.g., "sent_message"), it collects all preceding
    contextual events within a specified time window to create training samples
    that capture temporal relationships and sequential dependencies.

    The splitter includes intelligent filtering to maintain logical consistency in
    communication scenarios, preventing unrealistic temporal relationships such as
    receiving responses at the same timestamp as sending messages.

    The splitter is particularly useful for training models on sequential decision-making
    tasks where historical context influences future actions, such as autonomous agent
    communication in rescue operations.

    :inherits: :class:`~src.dataset.base.splitters.BaseSpliter`

    Key Features:
        - **Time-based windowing**: Collects events within configurable time windows
        - **Target-focused splitting**: Centers windows around specific event types
        - **Chronological ordering**: Maintains temporal sequence in training data
        - **Logical filtering**: Excludes contradictory simultaneous events
        - **Efficient lookups**: Uses binary search for scalable window creation

    Attributes:
        target_column (str): Event type to use as learning targets
        learning_column_not_to_include_on_target_timestamp (str): Event type to exclude when occurring at same timestamp as target
        max_window_size (int): Maximum time span (in timestamp units) for context windows

    Example:
        .. code-block:: python

            # Create splitter for message prediction
            splitter = SlicingWindowSplitter(
                target_column="sent_message",
                max_window_size=100  # 100 timestamp units lookback
            )

            # Split timeline data
            samples = splitter(timeline_data)

            # Each sample contains:
            # - Learning data: contextual events before target
            # - Target data: the message to predict
            for sample in samples:
                print(f"Context length: {sample.rollout_length}")
                print(f"Target: {sample.target_data}")

    Window Creation Process:
        1. **Separate events**: Split timeline into learning and target events
        2. **Sort by time**: Ensure chronological order for context
        3. **Define windows**: For each target, create time-bounded context window (inclusive of target timestamp)
        4. **Apply filtering**: Remove specified event types at target timestamp to maintain logical consistency
        5. **Filter empty**: Skip targets with insufficient context after filtering
        6. **Post-process**: Convert to structured training samples

    .. seealso::
        :class:`~src.dataset.base.splitters.BaseSpliter`
            Base splitter interface with generic typing
        :class:`~src.dataset.base.models.PostProcessedSample`
            Output format for processed training samples
    """

    def __init__(
        self,
        target_column: str = "sent_message",
        learning_column_not_to_include_on_target_timestamp: str = "recieved_message",
        max_window_size: int = 100,
        **kwargs,
    ):
        """
        Initialize the sliding window splitter.

        :param target_column: Type of events to use as prediction targets
        :type target_column: str
        :param learning_column_not_to_include_on_target_timestamp: Event type to exclude from learning context when it occurs at the same timestamp as target events (prevents logical inconsistencies like receiving responses before sending messages)
        :type learning_column_not_to_include_on_target_timestamp: str
        :param max_window_size: Maximum time span for context windows
        :type max_window_size: int
        :param kwargs: Additional configuration parameters for base splitter
        :type kwargs: dict

        Window Size Guidelines:
            - **Small windows (10-50)**: Focus on immediate context
            - **Medium windows (50-200)**: Capture short-term patterns
            - **Large windows (200+)**: Include long-term dependencies

        Temporal Filtering:
            The `learning_column_not_to_include_on_target_timestamp` parameter prevents
            logically inconsistent training data. For example, when predicting sent
            messages, received messages at the same timestamp are excluded since an
            agent cannot receive a response to a message it hasn't sent yet.

        Example:
            .. code-block:: python

                # Short-term context splitter
                short_splitter = SlicingWindowSplitter(
                    target_column="sent_message",
                    learning_column_not_to_include_on_target_timestamp="received_message",
                    max_window_size=50
                )

                # Long-term context splitter
                long_splitter = SlicingWindowSplitter(
                    target_column="sent_message",
                    learning_column_not_to_include_on_target_timestamp="received_message",
                    max_window_size=200
                )
        """
        super().__init__(**kwargs)
        self.target_column = target_column
        self.max_window_size = max_window_size
        self.learning_column_not_to_include_on_target_timestamp = (
            learning_column_not_to_include_on_target_timestamp
        )

    def post_process(self, splitted_data: SplittedData) -> list[PostProcessedSample]:
        """
        Convert split timeline data into structured training samples.

        Transforms the raw window-target pairs into formatted training samples
        by concatenating learning events into input text and extracting metadata
        for training management and evaluation.

        :param splitted_data: List of (learning_events, target_event) pairs
        :type splitted_data: SplittedData

        :returns: List of structured training samples with metadata
        :rtype: list[PostProcessedSample]

        Processing Steps:
            1. **Concatenate learning data**: Join formatted strings with newlines
            2. **Extract target data**: Use formatted target event content
            3. **Calculate metadata**: Determine timestamps and sequence length
            4. **Create samples**: Build PostProcessedSample objects

        Sample Structure:
            Each returned sample contains:
                - **learning_data**: Multi-line string of contextual events
                - **target_data**: Single target event content
                - **rollout_length**: Number of context events
                - **timestamps**: Start, end, and target timestamps

        Example:
            .. code-block:: python

                # Input: [(learning_events, target_event), ...]
                splitted = [
                    (
                        [event1, event2, event3],  # learning context
                        target_event               # prediction target
                    )
                ]

                # Output: [PostProcessedSample(...)]
                samples = splitter.post_process(splitted)

                sample = samples[0]
                print(sample.learning_data)
                # Output:
                # <T+0> <POS> 10 20
                # <T+5> <RCV> <AGENT#2> Status update
                # <T+3> <POS> 12 18

                print(sample.target_data)
                # Output: <SND> <TO> <AGENT#2> Acknowledged
        """
        post_processed = []
        for learning_data, target_data in splitted_data:
            post_processed.append(
                PostProcessedSample(
                    learning_data="\n".join(td.formatted for td in learning_data),
                    target_data=target_data.formatted,
                    rollout_length=len(learning_data),
                    start_timestamp=sorted([td.timestamp for td in learning_data])[0],
                    end_timestamp=sorted([td.timestamp for td in learning_data])[-1],
                    target_timestamp=target_data.timestamp,
                )
            )
        return post_processed

    def split(self, timeline_data: list[TimelineData]) -> SplittedData:
        """
        Split mixed timeline data into training windows and target events.

        Creates temporal training windows by pairing each target event with
        all contextual events within the specified time window, including events
        at the same timestamp as the target. Applies intelligent filtering to
        exclude logically inconsistent simultaneous events.

        :param timeline_data: Mixed list of learning and target events
        :type timeline_data: list[TimelineData]

        :returns: List of (context_window, target_event) pairs
        :rtype: SplittedData

        :raises ValueError: If timeline_data is empty or contains invalid timestamps

        Algorithm:
            1. **Separate events**: Split by event type (learning vs target)
            2. **Sort learning events**: Order by timestamp for binary search
            3. **For each target event**:
                - Calculate time window bounds: [target_time - max_window_size, target_time]
                - Use binary search to find events in window (inclusive of target timestamp)
                - Filter out specified event types at target timestamp to maintain logical consistency
                - Skip targets with empty context windows after filtering
            4. **Return pairs**: (filtered_context_events, target_event) tuples

        Time Complexity:
            - **Setup**: O(n log n) for sorting learning events
            - **Per target**: O(log n) binary search + O(k) window extraction + O(k) filtering
            - **Overall**: O(n log n + m log n) where n=learning events, m=target events

        Temporal Filtering:
            Events of type `learning_column_not_to_include_on_target_timestamp` are
            excluded when they occur at the same timestamp as the target event.
            This prevents training on logically impossible scenarios, such as an
            agent receiving a response at the exact moment it sends a message.

        Example:
            .. code-block:: python

                timeline = [
                    TimelineData(timestamp=100, type="position", formatted="<POS> 10 20"),
                    TimelineData(timestamp=120, type="received_message", formatted="<RCV> Update"),
                    TimelineData(timestamp=130, type="position", formatted="<POS> 12 18"),
                    TimelineData(timestamp=130, type="received_message", formatted="<RCV> Response"),  # Same timestamp
                    TimelineData(timestamp=130, type="sent_message", formatted="<SND> Roger"),  # Target
                    TimelineData(timestamp=150, type="sent_message", formatted="<SND> Moving"),
                ]

                # With max_window_size=40 and learning_column_not_to_include_on_target_timestamp="received_message"
                windows = splitter.split(timeline)

                # Results in:
                # [
                #   ([position_event@100, received_event@120, position_event@130], sent_message@130),
                #   # Note: received_message@130 is excluded from first target's context
                #   ([position_event@130], sent_message@150)  # only recent events
                # ]

        .. note::
            The algorithm ensures that context events are always temporally
            ordered and that no target event includes itself in its context window.
            Events at the same timestamp as targets are included unless they match
            the filtered event type.

        .. warning::
            Targets with no preceding events in the time window are automatically
            skipped to avoid training on empty contexts.
        """
        x_target_samples = []
        target_timeline_data = []
        learning_timeline_data = []

        for td in timeline_data:
            if td.type == self.target_column:
                target_timeline_data.append(td)
            else:
                learning_timeline_data.append(td)

        learning_timeline_data.sort(key=lambda td: td.timestamp)

        all_target_timestamp = [(d.timestamp, d) for d in target_timeline_data]
        all_learning_timestamps = [d.timestamp for d in learning_timeline_data]

        for target_t, target_v in all_target_timestamp:
            lower_bound = target_t - self.max_window_size
            if lower_bound < 0:
                lower_bound = 0

            i = bisect_left(all_learning_timestamps, lower_bound)
            j = bisect_left(all_learning_timestamps, target_t + 1)

            samples = learning_timeline_data[i:j]

            # Filter out learning events that shouldn't be included at target timestamp
            filtered_samples = []
            for sample in samples:
                if (
                    sample.type
                    == self.learning_column_not_to_include_on_target_timestamp
                    and sample.timestamp == target_t
                ):
                    continue
                filtered_samples.append(sample)

            if not filtered_samples:
                continue

            x_target_samples.append((filtered_samples, target_v))

        return x_target_samples
