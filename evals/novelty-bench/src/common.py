from openai import AsyncOpenAI

def oai_client():
    api_key=os.environ["OPENAI_API_KEY"]
    #awith open("openai-api-key") as file:
    return AsyncOpenAI(api_key=api_key)

