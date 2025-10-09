import dataclasses
from typing import List, Literal

MessageRole = Literal["system", "user", "assistant"]

@dataclasses.dataclass()
class Message:
    role: MessageRole
    content: str

    def to_dict(self):
        return dataclasses.asdict(self)

@dataclasses.dataclass
class Status:
    started: int = 0
    in_progress: int = 0
    succeeded: int = 0
    failed: int = 0
