from mistralai import Mistral
from mistralai.async_client import MistralAsyncClient


def create_client(api_key):
    """
    Create a Mistral client

    Args:
        api_key (str): Mistral AI API key

    Returns:
        client (obj): Mistral client
    """

    return Mistral(api_key)


def create_async_client(api_key):
    """
    Create a Mistral client

    Args:
        api_key (str): Mistral AI API key

    Returns:
        client (obj): Mistral client
    """

    return MistralAsyncClient(api_key)


def prompt_mistral(client, prompt, model="mistral-small-latest"):
    """
    Send a prompt to Mistral API and get a response

    Args:
        client (obj): Mistral client
        prompt (str): Question to ask, or prompt to send
        model (str): Model to use

    Returns:
        str: Model's response
    """

    messages = [{"role": "user", "content": prompt}]

    try:
        response = client.chat.complete(model=model, messages=messages)
        return response.choices[0].message.content
    except Exception:
        print(f"An error happened processing prompt: {prompt}")
        return ""


async def async_prompt_mistral(client, prompt, model="mistral-small-latest"):
    """
    Send a prompt to Mistral API and get a response asynchronously

    Args:
        client (obj): Async Mistral client
        prompt (str): Question to ask, or prompt to send
        model (str): Model to use

    Returns:
        str: Model's response
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        response = await client.chat(model=model, messages=messages)
        return response.choices[0].message.content
    except Exception:
        print(f"An error happened processing prompt: {prompt}")
        return ""
