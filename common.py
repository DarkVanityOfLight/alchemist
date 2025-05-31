from typing import Literal
from dataclasses import dataclass

type SetType = Literal["NATURALS", "INTEGERS", "POSITIVES", "REALS", "EMPTY"]

@dataclass(frozen=True)
class Variable:
    name: str
    domain: SetType

    def __repr__(self):
        return f"{self.name}: {self.domain}"
