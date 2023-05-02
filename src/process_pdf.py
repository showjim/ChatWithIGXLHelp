import os, re
from pathlib import Path
from llama_index import download_loader
from llama_index import (
    GPTSimpleVectorIndex,
)
from src.initial_llm import service_context

MAX_TEXT_INLINE = 100

def trim_text(text):
    """
    Trim text
    @param text:
    @return:
    """
    text = text.strip()
    text = re.sub('\s+', ' ', text)
    text = re.sub('\n+', '\n', text)

    return text


def limit_line_length(text):
    """
    Limit line length
    @param text:
    @return:
    """
    lines = []
    for line in text.split('\n'):
        chunks = [line[i:i+MAX_TEXT_INLINE] for i in range(0, len(line), MAX_TEXT_INLINE)]
        lines.extend(chunks)
    return '\n'.join(lines)

def get_all_files(filepath, ext_name):
    result_list = []
    for parent, dirnames, filenames in os.walk(filepath):
        for filename in filenames:
            if filename.endswith(ext_name): # and filename.startswith("TSB_"):
                result_list.append(parent + "/" + filename)
    return result_list

def create_index(UPLOAD_FOLDER, UPLOAD_FILE_NAME):
    INDEX_FILE = "index_" + UPLOAD_FILE_NAME + ".index"
    WORK_FOLDER_DIR = UPLOAD_FOLDER
    # setup PDF loader
    PDFReader = download_loader("CJKPDFReader")
    loader = PDFReader()

    cn_pdfs = get_all_files(WORK_FOLDER_DIR, "pdf")
    # documents = []
    file_path = os.path.join(Path(WORK_FOLDER_DIR), Path(UPLOAD_FILE_NAME))
    index_file = os.path.join(Path(WORK_FOLDER_DIR), Path(INDEX_FILE))
    # index_file = '.\Data\index_DIB_Design_Guideline.pdf.index' #os.path.join(Path(WORK_DIR), Path(INDEX_FILE))
    if os.path.exists(index_file) == False:
        # for filename in cn_pdfs:
        #     single_pdf = loader.load_data(filename)
        #     single_pdf[0].text = single_pdf[0].text.replace("。", ". ")
        #     documents.append(single_pdf[0])
        # documents = SimpleDirectoryReader(WORK_FOLDER_DIR).load_data()
        documents = loader.load_data(file_path)
        # documents[0].text = documents[0].text.replace("。", ". ")
        documents[0].text = limit_line_length(trim_text(documents[0].text))
        index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
        index.save_to_disk(index_file)
    else:
        index = GPTSimpleVectorIndex.load_from_disk(index_file, service_context=service_context)
    return index
