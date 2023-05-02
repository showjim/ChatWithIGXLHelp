from src.initial_llm import llm_predictor, service_context
from llama_index.prompts.prompts import QuestionAnswerPrompt
def chat(index, query):
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