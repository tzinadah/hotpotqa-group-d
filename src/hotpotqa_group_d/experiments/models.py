"""
This expiremnt is to try different models than the baseline which
used the small-latest mistral model
"""

from hotpotqa_group_d.config import Model
from hotpotqa_group_d.pipelines import basic_answer

if __name__ == "__main__":
    # Medium model
    basic_answer("results/medium-model.json", model=Model.MEDIUM, sample_size=100)

    # Large model
    basic_answer("results/large-model.json", model=Model.LARGE, sample_size=100)
