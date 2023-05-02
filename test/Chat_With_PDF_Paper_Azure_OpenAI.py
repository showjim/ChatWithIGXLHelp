import os, json, PyPDF2
import openai
import shutil
from dotenv import load_dotenv
from pathlib import Path
from IPython.display import Markdown, display
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index import LangchainEmbedding
from llama_index import (
    GPTSimpleVectorIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    PromptHelper,
    ServiceContext
)
from llama_index.prompts.prompts import QuestionAnswerPrompt
from llama_index.response.notebook_utils import display_response

WORK_ENV_DIR = './'
WORK_FOLDER_DIR = './Data/'
INDEX_FILE = 'index_IG-XL_Helps.index'
ENV_FILE = './key.txt'

shutil.copyfile(os.path.join(WORK_ENV_DIR, ENV_FILE), "./.env")
load_dotenv()
# Load config values
with open(r'./config.json') as config_file:
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
    query_model_name=embedding_model_name,
    #embed_batch_size=1 # this is special for Azure, incase error openai:'Too many inputs for model None. The max number of inputs is 1.
))

# max LLM token input size
max_input_size = 4096
# set number of output tokens
num_output = 1024#512
# set maximum chunk overlap
max_chunk_overlap = 20

prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap, chunk_size_limit=2048)
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    embed_model=embedding_llm,
    prompt_helper=prompt_helper
)

index_file = os.path.join(Path(WORK_FOLDER_DIR), Path(INDEX_FILE))
# index_file = '.\Data\index_DIB_Design_Guideline.pdf.index' #os.path.join(Path(WORK_DIR), Path(INDEX_FILE))
if os.path.exists(index_file) == False:
    documents = SimpleDirectoryReader(WORK_FOLDER_DIR).load_data()
    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
    index.save_to_disk(index_file)
else:
    index = GPTSimpleVectorIndex.load_from_disk(index_file, service_context=service_context)


def chat(query):
    QUESTION_ANSWER_PROMPT_TMPL = """
    PRETEND YOU ARE GPT-4 MODEL. You are given the following extracted parts of a long document and a question. Provide a conversational answer based on the context provided.
    If you can't find the answer in the context below, just say "Hmm, I'm not sure." Don't try to make up an answer.
    If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
    Context information is below.
    =========
    {context_str}
    =========
    {query_str}
    """
    QUESTION_ANSWER_PROMPT = QuestionAnswerPrompt(QUESTION_ANSWER_PROMPT_TMPL)
    result = index.query(
      query,
      service_context=service_context,
      text_qa_template=QUESTION_ANSWER_PROMPT,
      # default: For the given index, “create and refine” an answer by sequentially
      #   going through each Node; make a separate LLM call per Node. Good for more
      #   detailed answers.
      # compact: For the given index, “compact” the prompt during each LLM call
      #   by stuffing as many Node text chunks that can fit within the maximum prompt size.
      #   If there are too many chunks to stuff in one prompt, “create and refine” an answer
      #   by going through multiple prompts.
      # tree_summarize: Given a set of Nodes and the query, recursively construct a
      #   tree and return the root node as the response. Good for summarization purposes.
      response_mode="tree_summarize",
      similarity_top_k=3,
      # mode="default" will a create and refine an answer sequentially through
      #   the nodes of the list.
      # mode="embedding" will synthesize an answer by
      #   fetching the top-k nodes by embedding similarity.
      mode="embedding",
    )
    print(f"Token used: {llm_predictor.last_token_usage}, total used: {llm_predictor.total_tokens_used}")
    return result

# resp = chat('What is the pin fan-out?')
# resp = chat('What is UI stands for in jitter test?')

# print(resp.get_formatted_sources())
# print('answer was:', resp.response)
# resp.print_response_stream()
# display_response(resp)


