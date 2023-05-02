import os, json, PyPDF2
import openai
import shutil
from pathlib import Path
from dotenv import load_dotenv
from IPython.display import Markdown, display
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index import LangchainEmbedding
from llama_index import (
    GPTTreeIndex,
    GPTSimpleVectorIndex,
    GPTSimpleKeywordTableIndex,
    GPTListIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    PromptHelper,
    ServiceContext
)
from llama_index.prompts.prompts import QuestionAnswerPrompt
from llama_index.indices.composability import ComposableGraph
from llama_index.response.notebook_utils import display_response
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform


WORK_ENV_DIR = './'
WORK_FOLDER_DIR = "./Data/"
INDEX_FILE = "index_IGXL_Helps_Outer_Graph.index" #"index_IGXL_Helps_Graph.index"
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
                      temperature=0.2,
                      )

llm_predictor = LLMPredictor(llm=llm)

# For outer_graph
decompose_transform = DecomposeQueryTransform(
    llm_predictor, verbose=True
)

embedding_llm = LangchainEmbedding(OpenAIEmbeddings(
    document_model_name=embedding_model_name,
    query_model_name=embedding_model_name,
    #embed_batch_size=1 # this is special for Azure, incase error openai:'Too many inputs for model None. The max number of inputs is 1.
))

# documents = SimpleDirectoryReader(WORK_FOLDER_DIR).load_data()

# max LLM token input size
max_input_size = 4096
# set number of output tokens
num_output = 512
# set maximum chunk overlap
max_chunk_overlap = 20

prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap, chunk_size_limit=2048)
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    embed_model=embedding_llm,
    prompt_helper=prompt_helper
)

# Process PDF files
# IGXL Help List
IGXL_HELPS = ['DIB_Design_Guideline','Support_Board','Specifications_UltraFLEXplus','UltraPin2200',
              'UltraVS64','UltraVS256-HP','UltraVS256-HP','UltraVI264', 'SCAN']
graph_file = os.path.join(Path(WORK_FOLDER_DIR), Path(INDEX_FILE))
# index_file = '.\Data\index_DIB_Design_Guideline.pdf.index' #os.path.join(Path(WORK_DIR), Path(INDEX_FILE))
if os.path.exists(graph_file) == False:
    igxl_help_indices = {}
    index_summaries = {}
    if os.path.exists(WORK_FOLDER_DIR + f'index_DIB_Design_Guideline.index') == False:
        # Load all igxl help documents
        igxl_help_pdfs = {}
        for igxl_help in IGXL_HELPS:
            igxl_help_pdfs[igxl_help] = SimpleDirectoryReader(input_files=[WORK_FOLDER_DIR + f"{igxl_help}.pdf"]).load_data()
        # Build igxl help document index
        for igxl_help in IGXL_HELPS:
            igxl_help_indices[igxl_help] = GPTSimpleVectorIndex.from_documents(igxl_help_pdfs[igxl_help], service_context=service_context)
            # igxl_help_indices[igxl_help] = GPTTreeIndex.from_documents(igxl_help_pdfs[igxl_help], service_context=service_context)
            # set summary text for each igxl help pdf
            index_summaries[igxl_help] = f"IGXL help documents about {igxl_help}"
            igxl_help_indices[igxl_help].save_to_disk(WORK_FOLDER_DIR + f'index_{igxl_help}.index')
    else:
        for igxl_help in IGXL_HELPS:
            # set summary text for each igxl help pdf
            index_summaries[igxl_help] = (f"IGXL help documents about {igxl_help}"
                                          f"Use this index if you need to lookup specific facts about {igxl_help}.\n"
                                          "Do not use this index if you want to analyze multiple instruments."
                                          )
            igxl_help_indices[igxl_help] = GPTSimpleVectorIndex.load_from_disk(
                WORK_FOLDER_DIR + f'index_{igxl_help}.index',
                service_context = service_context
                )
    graph = ComposableGraph.from_indices(
        GPTSimpleKeywordTableIndex, #GPTListIndex,#GPTSimpleKeywordTableIndex,
        [index for _, index in igxl_help_indices.items()],
        [summary for _, summary in index_summaries.items()],
        service_context=service_context,
        max_keywords_per_chunk=100
        )

    # get root index
    root_index = graph.get_index(graph.index_struct.root_id, GPTSimpleKeywordTableIndex)
    # set id of root index
    root_index.index_struct.index_id = "compare_contrast"
    root_summary = (
        "This index contains IG-XL articles about multiple instruments. "
        "Use this index if you want to compare multiple instruments. "
    )

    # num children is num vector indexes + graph
    num_children = len(igxl_help_indices) + 1
    outer_graph = ComposableGraph.from_indices(
        GPTTreeIndex,
        [index for _, index in igxl_help_indices.items()] + [root_index],
        [summary for _, summary in index_summaries.items()] + [root_summary],
        num_children=num_children
    )

    # [optional] save to disk
    outer_graph.save_to_disk(graph_file)

else:
    print(graph_file)
    outer_graph = ComposableGraph.load_from_disk(graph_file, service_context=service_context)


def chat(query):
    QUESTION_ANSWER_PROMPT_TMPL = """
    You are an AI assistant providing helpful advice. You are given the following extracted parts of a long document and a question. Provide a conversational answer based on the context provided.
    If you can't find the answer in the context below, just say "Hmm, I'm not sure." Don't try to make up an answer.
    If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
    Context information is below.
    =========
    {context_str}
    =========
    {query_str}
    """
    QUESTION_ANSWER_PROMPT = QuestionAnswerPrompt(QUESTION_ANSWER_PROMPT_TMPL)

    # set query config
    # query_configs = [
    #     {
    #         "index_struct_type": "simple_dict",
    #         "query_mode": "default",
    #         "query_kwargs": {
    #             "similarity_top_k": 3,
    #             "response_mode": "tree_summarize"
    #         }
    #     },
    #     {
    #         "index_struct_type": "list",
    #         "query_mode": "default",
    #         "query_kwargs": {
    #             "response_mode": "tree_summarize",
    #             "use_async": True,
    #             "verbose": True
    #         },
    #     },
    # ]

    # set query config
    query_configs = [
        {
            "index_struct_type": "keyword_table",
            "query_mode": "simple",
            "query_kwargs": {
                "response_mode": "tree_summarize",
                "verbose": True
            },
        },
        {
            "index_struct_type": "tree",
            "query_mode": "default", #default

        }
    ]
    for igxl_help in IGXL_HELPS:
        query_config = {
            "index_struct_id": igxl_help,
            "index_struct_type": "simple_dict",
            "query_mode": "embedding", #default
            "query_kwargs": {
                "similarity_top_k": 3
            },
            # NOTE: set query transform for subindices
            "query_transform": decompose_transform
        }
        query_configs.append(query_config)

    result = outer_graph.query(
        query,
        # nop service_context&query_configs, then I can make AI get answer, otherwise AI can not find relavent context
        service_context=service_context,
        query_configs=query_configs,
    )
    print(f"Token used: {llm_predictor.last_token_usage}, total used: {llm_predictor.total_tokens_used}")
    return result

resp = chat('Compare and contrast UltraVI264 and UltraVS64, and list the results.')
# resp = chat('What is the channel merge rules of UltraVI264.')

print(resp.get_formatted_sources())
print('answer was:', resp)
# display_response(resp)


