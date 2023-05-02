import os, json
import openai
import shutil
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index import LangchainEmbedding
from llama_index import (
    LLMPredictor,
    PromptHelper,
    ServiceContext
)

WORK_ENV_DIR = './'
WORK_FOLDER_DIR = './登幽州台歌/'
INDEX_FILE = 'index_poem.index'
ENV_FILE = 'key.txt'

shutil.copyfile(os.path.join(WORK_ENV_DIR, ENV_FILE), ".env")
load_dotenv()
# Load config values
with open(r'config.json') as config_file:
    config_details = json.load(config_file)

# Setting up the embedding model
embedding_model_name = config_details['EMBEDDING_MODEL']
openai.api_type = "azure"
openai.api_base = config_details['OPENAI_API_BASE']
openai.api_version = config_details['EMBEDDING_MODEL_VERSION']
openai.api_key = os.getenv("OPENAI_API_KEY")

## Setting up the chat model
# llm = AzureOpenAI(deployment_name=config_details['CHATGPT_MODEL'], model_kwargs={
#     "api_key": openai.api_key,
#     "api_base": openai.api_base,
#     "api_type": openai.api_type,
#     "api_version": config_details['OPENAI_API_VERSION'],
# })

llm = AzureChatOpenAI(deployment_name=config_details['CHATGPT_MODEL'],
                      openai_api_key=openai.api_key,
                      openai_api_base=openai.api_base,
                      openai_api_type =openai.api_type,
                      openai_api_version=config_details['OPENAI_API_VERSION'],
                        temperature=0.5,
                      )

llm_predictor = LLMPredictor(llm=llm)

embedding_llm = LangchainEmbedding(OpenAIEmbeddings(
    document_model_name=embedding_model_name,
    query_model_name=embedding_model_name
    ),
    embed_batch_size=1 # this is special for Azure, incase error openai:'Too many inputs for model None. The max number of inputs is 1.
)

# max LLM token input size
max_input_size = 3900 #4096
# set number of output tokens
num_output = 512#512
# set maximum chunk overlap
max_chunk_overlap = 20

prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap, chunk_size_limit=1024)
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    embed_model=embedding_llm,
    prompt_helper=prompt_helper,
    chunk_size_limit=1024
)
