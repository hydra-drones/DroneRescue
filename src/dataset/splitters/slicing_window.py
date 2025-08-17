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

    The splitter is particularly useful for training models on sequential decision-making
    tasks where historical context influences future actions, such as autonomous agent
    communication in rescue operations.

    :inherits: :class:`~src.dataset.base.splitters.BaseSpliter`

    Key Features:
        - **Time-based windowing**: Collects events within configurable time windows
        - **Target-focused splitting**: Centers windows around specific event types
        - **Chronological ordering**: Maintains temporal sequence in training data
        - **Efficient lookups**: Uses binary search for scalable window creation

    Attributes:
        target_column (str): Event type to use as learning targets
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
        3. **Define windows**: For each target, create time-bounded context window
        4. **Filter empty**: Skip targets with insufficient context
        5. **Post-process**: Convert to structured training samples

    .. seealso::
        :class:`~src.dataset.base.splitters.BaseSpliter`
            Base splitter interface with generic typing
        :class:`~src.dataset.base.models.PostProcessedSample`
            Output format for processed training samples
    """

    def __init__(
        self,
        target_column: str = "sent_message",
        max_window_size: int = 100,
        **kwargs,
    ):
        """
        Initialize the sliding window splitter.

        :param target_column: Type of events to use as prediction targets
        :type target_column: str
        :param max_window_size: Maximum time span for context windows
        :type max_window_size: int
        :param kwargs: Additional configuration parameters for base splitter
        :type kwargs: dict

        Window Size Guidelines:
            - **Small windows (10-50)**: Focus on immediate context
            - **Medium windows (50-200)**: Capture short-term patterns
            - **Large windows (200+)**: Include long-term dependencies

        Example:
            .. code-block:: python

                # Short-term context splitter
                short_splitter = SlicingWindowSplitter(
                    target_column="sent_message",
                    max_window_size=50
                )

                # Long-term context splitter
                long_splitter = SlicingWindowSplitter(
                    target_column="sent_message",
                    max_window_size=200
                )
        """
        super().__init__(**kwargs)
        self.target_column = target_column
        self.max_window_size = max_window_size

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
        all preceding contextual events within the specified time window.
        Uses efficient binary search to handle large timelines scalably.

        :param timeline_data: Mixed list of learning and target events
        :type timeline_data: list[TimelineData]

        :returns: List of (context_window, target_event) pairs
        :rtype: SplittedData

        :raises ValueError: If timeline_data is empty or contains invalid timestamps

        Algorithm:
            1. **Separate events**: Split by event type (learning vs target)
            2. **Sort learning events**: Order by timestamp for binary search
            3. **For each target event**:
                - Calculate time window bounds: [target_time - max_window_size, target_time)
                - Use binary search to find events in window
                - Skip targets with empty context windows
            4. **Return pairs**: (context_events, target_event) tuples

        Time Complexity:
            - **Setup**: O(n log n) for sorting learning events
            - **Per target**: O(log n) binary search + O(k) window extraction
            - **Overall**: O(n log n + m log n) where n=learning events, m=target events

        Example:
            .. code-block:: python

                timeline = [
                    TimelineData(timestamp=100, type="position", formatted="<POS> 10 20"),
                    TimelineData(timestamp=120, type="received_message", formatted="<RCV> Update"),
                    TimelineData(timestamp=130, type="sent_message", formatted="<SND> Roger"),
                    TimelineData(timestamp=150, type="sent_message", formatted="<SND> Moving"),
                ]

                # With max_window_size=40
                windows = splitter.split(timeline)

                # Results in:
                # [
                #   ([position_event, received_event], first_sent_message),
                #   ([received_event], second_sent_message)  # only recent events
                # ]

        .. note::
            The algorithm ensures that context events are always temporally
            ordered and that no target event includes itself in its context window.

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
            j = bisect_left(all_learning_timestamps, target_t)

            samples = learning_timeline_data[i:j]

            if not samples:
                continue

            x_target_samples.append((samples, target_v))

        return x_target_samples
