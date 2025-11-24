def vanilla_template(prompt):
    """
    Return the question with extra simple instructions

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


def expert_prompting(prompt):
    """
    Ask the model to act like an expert in the field (trivia in this case)

    Args:
        prompt (str): Original prompt

    Returns:
        formatted_prompt (str): Prompt after applying the template
    """

    formatted_prompt = vanilla_template(prompt)
    formatted_prompt = f"""
    You are a trivia master that is really good at ansering random trivia questions. You are practically
    an encyclopedia use all those skills and knowledge to answer the following quesion.
    {formatted_prompt}
    """
    return formatted_prompt
