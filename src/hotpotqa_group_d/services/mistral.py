from mistralai import Mistral


def prompt_mistral(api_key, prompt, model="mistral-small-latest"):
    """
    Send a prompt to Mistral API and get a response

    Args:
        api_key (str): Mistral API key
        prompt (str): The question to ask, or prompt to send
        model (str): The model to use

    Returns:
        str: The model's response
    """

    client = Mistral(api_key)

    messages = [{"role": "user", "content": prompt}]
    response = client.chat.complete(model=model, messages=messages)

    return response.choices[0].message.content
