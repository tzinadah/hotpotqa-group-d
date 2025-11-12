import json


def parse_data(file_path="data/hotpot_dev_fullwiki_v1"):
    """
    Function that parses the hotpotqa data and gets the questions out of them

    Args:
        file_path (str): path to the data file

    Returns:
        questions (dict): dictionary of the hotpotqa data
    """

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    print(data[0:3])
    return data
