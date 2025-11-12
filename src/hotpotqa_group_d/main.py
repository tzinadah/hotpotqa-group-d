from hotpotqa_group_d.config import Env
from hotpotqa_group_d.services import (
    create_client,
    format_results,
    parse_data,
    prompt_mistral,
)

if __name__ == "__main__":

    # Simple answers for baseline evaluation
    env = Env()
    client = create_client(env.MISTRAL_KEY)
    dev_fullwiki_data = parse_data()

    # Results
    qa_pairs = list()

    for idx, data_point in enumerate(dev_fullwiki_data):
        # Progress tracking
        print(f"asnwering {idx + 1}/{len(dev_fullwiki_data)}")

        question = data_point["question"]
        answer = prompt_mistral(client, question)
        qa_pairs.append((question, answer))

    format_results(qa_pairs, file_path="results/baseline.json")
