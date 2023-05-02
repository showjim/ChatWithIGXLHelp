import os, json
import openai
import shutil
from dotenv import load_dotenv
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index import LangchainEmbedding
from llama_index import (
    GPTSimpleVectorIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    PromptHelper,
    ServiceContext
)

WORK_DIR = '../'
ENV_FILE = '../key.txt'

shutil.copyfile(os.path.join(WORK_DIR, ENV_FILE), "../.env")
load_dotenv()
# Load config values
with open(r'../config.json') as config_file:
    config_details = json.load(config_file)

# Setting up the env
model_name = config_details['CHATGPT_MODEL']
openai.api_type = "azure"
openai.api_base = config_details['OPENAI_API_BASE']
openai.api_version = config_details['OPENAI_API_VERSION']
openai.api_key = os.getenv("OPENAI_API_KEY")
# import time
# start_t = time.localtime(int('1681817823'))
# print(str(time.strftime("%Y/%m/%d-%H:%M:%S", start_t)))

# setup chat
max_response_tokens = 512

def chat_test():
    user_message = "I want to write a blog post about the impact of AI on the future of work."
    response = openai.ChatCompletion.create(
        engine=model_name, # engine = "deployment_name".
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            # {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
            # {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=max_response_tokens,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )


    print(response)
    print("==================================")
    print(response['choices'][0]['message']['content'])
    print("==================================")

def chat_loop():
    conversation=[{"role": "system", "content": "You are a helpful assistant."}]
    while(True):
        user_input = input()
        conversation.append({"role": "user", "content": user_input})

        response = openai.ChatCompletion.create(
            engine=model_name, # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
            messages = conversation
        )

        conversation.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
        print("\n" + response['choices'][0]['message']['content'] + "\n")