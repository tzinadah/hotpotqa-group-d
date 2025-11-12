import json


def parse_data(file_path="data/hotpot_dev_fullwiki_v1.json"):
    """
    Function that parses the hotpotqa data and gets the questions out of them

    Args:
        file_path (str): path to the data file

    Returns:
        questions (dict): dictionary of the hotpotqa data
    """

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def format_results(qa_pairs, file_path):
    """
    Function that takes the questions answer pairs formats
    them and writes them to a result file under the specified path

    Args:
        qa_pairs (list[(str, str)]): List of in the form of (question, answer)
        file_path (str): name of file to write to
    """

    # Formatted data to be written in the json file
    data = dict()
    data["answer"] = dict()

    for question, answer in qa_pairs:
        data["answer"][question] = answer

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file)
