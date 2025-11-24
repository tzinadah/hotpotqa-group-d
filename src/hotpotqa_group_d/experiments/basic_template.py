"""
This expiremnt is to try a basic template asking the model to be concise
and to instruct it not to explain itself.

The model used is medium-latest since the larger model doesn't show that much of an improvment
keeping in mind costs and performance.
"""

from hotpotqa_group_d.config import Model, vanilla_template
from hotpotqa_group_d.pipelines.answering_pipelines import templated_answer

if __name__ == "__main__":
    templated_answer(
        result_path="results/basic-template.json",
        template=vanilla_template,
        model=Model.MEDIUM,
        sample_size=100,
    )
