from mistralai import Mistral


def create_client(api_key):
    """
    Create a Mistral client

    Args:
        api_key (str): Mistral AI API key

    Returns:
        client (obj): Mistral client
    """

    return Mistral(api_key)


def prompt_mistral(client, prompt, model="mistral-small-latest"):
    """
    Send a prompt to Mistral API and get a response

    Args:
        client (obj): Mistral client
        prompt (str): The question to ask, or prompt to send
        model (str): The model to use

    Returns:
        str: The model's response
    """

    messages = [{"role": "user", "content": prompt}]
    response = client.chat.complete(model=model, messages=messages)

    return response.choices[0].message.content
