import asyncio

from hotpotqa_group_d.config import Env
from hotpotqa_group_d.services import (
    async_prompt_mistral,
    create_client,
    format_results,
    parse_data,
    prompt_mistral,
)


def basic_answer(result_path):
    """
    Simple answers for baseline evaluation

    Args:
        result_path (str): File path to write results
    """

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

    # Filter out errored results
    successful_pairs = [pair for pair in qa_pairs if pair[1] != ""]

    format_results(qa_pairs, file_path=result_path)


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
        tasks.append(async_prompt_mistral(client, question))

    # Run all tasks in parallel
    print("Answering questions in parallel")
    qa_pairs = await asyncio.gather(*tasks)

    # Filter out errored results
    successful_pairs = [pair for pair in qa_pairs if pair[1] != ""]

    format_results(successful_pairs, file_path="results/baseline_async.json")
