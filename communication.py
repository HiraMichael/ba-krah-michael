from mango import json_serializable
from dataclasses import dataclass
import numpy as np


@json_serializable
@dataclass
class LoadAndGenerationDataMsg:
    load_and_generation_data: dict


@json_serializable
@dataclass
class AllocationMsg:
    pass


@json_serializable
@dataclass
class NewStepMsg:
    time: int


@json_serializable
@dataclass
class CanonSpecificSharesMsg:
    pass


@json_serializable
@dataclass
class YourCanonSpecificShareMsg:
    canon_specific_shares: np.array


@json_serializable
@dataclass
class NewCanonWeightsMsg:
    new_canon_weights: np.array


@json_serializable
@dataclass
class MyCanonWeightsMsg:
    my_canon_weights: np.array
