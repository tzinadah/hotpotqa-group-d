def vanilla_template(prompt):
    """
    Return the question with extra simple instructions

    Args:
        prompt (str): Original prompt

    Returns:
        formatted_prompt (str): Prompt after applying the template
    """

    formatted_prompt = f"""
    The question is:
    ###
    {prompt}
    ###
    """

    return formatted_prompt


def clear_template(prompt):
    """
    Return the prompt alongside a clear instruction to tell the LLM what to do clearly

    Args:
        prompt (str): Original prompt

    Returns:
        formatted_prompt (str): Prompt after applying the template
    """

    formatted_prompt = f"""
    The answer you give must be correct concise and straight to the point. Do not provide explaination.
    
    {vanilla_template(prompt)}
    """

    return formatted_prompt


def expert_template(prompt):
    """
    Ask the model to act like an expert in the field (trivia in this case)

    Args:
        prompt (str): Original prompt

    Returns:
        formatted_prompt (str): Prompt after applying the template
    """

    formatted_prompt = f"""
    You are a trivia master that is really good at ansering random trivia questions. You are practically
    an encyclopedia use all those skills and knowledge to answer the following quesion.

    {vanilla_template(prompt)}
    """
    return formatted_prompt


def polite_template(prompt):
    """
    Ask the model with simple instructions but in a polite manner

    Args:
        prompt (str): Original prompt

    Returns:
        formatted_prompt (str): Prompt after applying the template
    """

    formatted_prompt = f"""
    Could you please answer the following question? 
    Please keep your response correct and concise, with just the answer and no added explanation.

    {vanilla_template(prompt)}
    """

    return formatted_prompt


def blackmail_template(prompt):
    """
    Ask the model with simple instructions but indicate that the question holds high
    stakes using emotional blackmail

    Args:
        prompt (str): Original prompt

    Returns:
        formatted_prompt (str): Prompt after applying the template
    """

    formatted_prompt = f"""
    Your mom is held at gunpoint. If you don't answer the following quesion correctly,
    she dies immediately. In order for her to survive you have to answer accurately and concisely. Do
    not provide any explaination as that will result in your answer being autmatically wrong.

    {vanilla_template(prompt)}
    """

    return formatted_prompt
