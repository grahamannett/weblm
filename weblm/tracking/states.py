from dataclasses import dataclass, KW_ONLY
import uuid


@dataclass
class AgentState:
    """Base class for all states."""

    objective: str
    url: str
    screenshot_path: str

    previous_state: "AgentState" = None
    next_state: "AgentState" = None

    _: KW_ONLY
    _id: uuid.UUID = None

    def __post_init__(self):
        if self.id_ is None:
            self.id_ = uuid.uuid4()

    @classmethod
    def from_previous(cls, previous_state: "AgentState", **kwargs):
        return cls(
            previous_state.objective,
            previous_state.url,
            previous_state.screenshot_path,
            previous_state=previous_state,
            **kwargs,
        )
