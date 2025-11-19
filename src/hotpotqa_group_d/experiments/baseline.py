from hotpotqa_group_d.config.consts import Model
from hotpotqa_group_d.pipelines import basic_answer

if __name__ == "__main__":
    basic_answer("results/baseline.json", model=Model.SMALL, sample_size=100)
