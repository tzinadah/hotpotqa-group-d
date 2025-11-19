import asyncio

from hotpotqa_group_d.config import Env
from hotpotqa_group_d.config.consts import Model
from hotpotqa_group_d.services import (
    async_prompt_mistral,
    create_client,
    format_results,
    parse_data,
    prompt_mistral,
)


def basic_answer(result_path, model=Model.SMALL, sample_size=None):
    """
    Simple answers for baseline evaluation

    Args:
        result_path (str): File path to write results
        model (str) : Name of the model used in the pipeline
        sample_size (int): Number of samples used for answering
    """

    env = Env()
    client = create_client(env.MISTRAL_KEY)
    dev_fullwiki_data = parse_data()

    if sample_size:
        dev_fullwiki_data = dev_fullwiki_data[0:sample_size]
        print(f"limited dataset size to: {len(dev_fullwiki_data)}")

    # Results
    qa_pairs = list()

    for idx, data_point in enumerate(dev_fullwiki_data):
        # Progress tracking
        print(f"answering {idx + 1}/{len(dev_fullwiki_data)}")
        question = data_point["question"]
        answer = prompt_mistral(client, question, model)
        qa_pairs.append((data_point["_id"], answer))

    # Filter out errored results
    successful_pairs = [pair for pair in qa_pairs if pair[1] != ""]

    format_results(successful_pairs, file_path=result_path)


async def async_answer(result_path):
    """
    Improvement on simple answer by making the calls async for better performance

    Args:
        result_path (str): File path to write results
    """

    env = Env()
    client = create_client(env.MISTRAL_KEY)
    dev_fullwiki_data = parse_data()

    # Create a list of concurrent tasks
    tasks = []
    for data_point in dev_fullwiki_data:
        question = data_point["question"]
        question_id = data_point["_id"]
        tasks.append(async_prompt_mistral(client, question, question_id))

    # Run all tasks in parallel
    print("Answering questions in parallel")
    qa_pairs = await asyncio.gather(*tasks)

    # Filter out errored results
    successful_pairs = [pair for pair in qa_pairs if pair[1] != ""]

    format_results(successful_pairs, file_path="results/baseline_async.json")
