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
    If possible keep the answer as short as possible. Keep the answer as raw text without colors or markdowns.
    
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

    {clear_template(prompt)}
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

    {clear_template(prompt)}
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

    {clear_template(prompt)}
    """

    return formatted_prompt


def reasoning_template(prompt):
    """
    Ask the model to revise and "think" about his answer

    Args:
        prompt (str): Original prompt

    Returns:
        formatted_prompt (str): Prompt after applying the template
    """

    formatted_prompt = f"""
    You are gonna be provided a question. Think in detail about that question. Break it down
    into steps. Finally, revise your answer to make sure it's consistent.

    {clear_template(prompt)}
    """

    return formatted_prompt


def RAG_template(prompt, context, template=clear_template):
    """
    Format the prompt alongside the retrieved context

    Args:
        prompt (str): Original prompt
        context (str): Context of docs

    Returns:
        formatted_prompt (str): Prompt after applying the template
    """

    formatted_prompt = f"""
    You are gonna be provided a context and a question on that context.
    Use ONLY the context to answer the question. Don't use any prior knowledge.
    
    Context:
    {context}

    {template(prompt)}
    """

    return formatted_prompt

def reflection_template(question, context):
    """
    Ask the model to rerank the context chunks by usefulness for answering the question.

    """

    formatted_prompt = f"""
    You are given a list of context chunks and a question.

    Your task is to:
    1. RERANK the context chunks from most useful to least useful for answering the question.
    2. Return ONLY the reranked context, in the SAME FORMAT as it was given to you (no explanations, no extra text).

    Context:
    {context}

    Question:
    {question}
    """
    return formatted_prompt