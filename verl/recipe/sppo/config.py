from dataclasses import dataclass

from verl.workers.config import FSDPActorConfig


@dataclass
class SPPOActorConfig(FSDPActorConfig):
    sppo_eta: float = 1.0
