from enum import Enum

MODELS = ["mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"]


class Model(Enum):
    SMALL = "mistral-small-latest"
    MEDIUM = "mistral-medium-latest"
    LARGE = "mistral-large-latest"
