def vanilla_template(prompt):
    """
    Return the question with extra simple instruction

    Args:
        prompt (str): Original prompt

    Returns:
        formatted_prompt (str): Prompt after applying the template
    """

    formatted_prompt = f"""
    The answer you give must be correct concise and straight to the point. Do not provide explaination.
    The question is:
    ###
    {prompt}
    ###
    """

    return formatted_prompt
