def basic_prompt_template(prompt):
    """
    Return the question with extra simple instruction

    Args:
        prompt (str): Original prompt

    Returns:
        formatted_prompt (str): Prompt after applying the template
    """

    formatted_prompt = f"""
    You are a trivia master and you are playing a current high stake game of trivia. The answer you give 
    must be correct concise and straight to the point. Do not provide explaination.
    The question is:
    ###
    {prompt}
    ###
    """

    return formatted_prompt
