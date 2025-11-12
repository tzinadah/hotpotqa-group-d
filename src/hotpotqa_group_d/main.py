from hotpotqa_group_d.config import Env
from hotpotqa_group_d.services.mistral import prompt_mistral

if __name__ == "__main__":
    env = Env()
    res = prompt_mistral(env.MISTRAL_KEY, "hi")
    print(res)
