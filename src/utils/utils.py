from datetime import datetime
from typing import Tuple, Type

from torch import nn


def get_current_datetime_string() -> str:
    """
    Returns the current datetime as a string in the format: YYYY-MM-DD_HH-MM-SS
    :return: datetime string
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_parameter_names(model: nn.Module, forbidden_layer_types: Tuple[Type[nn.Module]] = (nn.LayerNorm,)):
    """
    Returns the names of the models parameters that are not inside a forbidden layer.
    :param model: models to get the parameters from
    :param forbidden_layer_types: layer types that are forbidden to have parameters
    :return: list of parameter names
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add models specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result
