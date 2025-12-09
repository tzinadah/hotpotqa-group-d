from hotpotqa_group_d.config import Env, Model
from hotpotqa_group_d.services import create_client, reasoning_prompt_mistral

if __name__ == "__main__":
    env = Env()

    client = create_client(env.MISTRAL_KEY)

    print(reasoning_prompt_mistral(client, "Hi", Model.REASONING))
