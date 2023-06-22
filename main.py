import os

import openai
import torch
from langchain.chat_models import ChatOpenAI
from llama_index import download_loader, GPTVectorStoreIndex, PromptHelper, LLMPredictor, ServiceContext, ListIndex, \
    StorageContext, load_index_from_storage, Document
import gradio as gr
from pathlib import Path
from llama_index.readers.qdrant import QdrantReader
from dotenv import load_dotenv
from InstructorEmbedding import INSTRUCTOR
from qdrant_client import QdrantClient

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]


def create_service_context():
    # constraint parameters
    max_input_size = 4096
    num_outputs = 3072
    max_chunk_overlap = .2
    chunk_size_limit = 600

    # allows the user to explicitly set certain constraint parameters
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # LLMPredictor is a wrapper class around LangChain's LLMChain that allows easy integration into LlamaIndex
    # llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.5, model_name="gpt-4", max_tokens=num_outputs))
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=1.0, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    # constructs service_context
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    return service_context


UnstructuredReader = download_loader("UnstructuredReader", refresh_cache=True)
loader = UnstructuredReader()


def data_ingestion_indexing():
    # https://pmarca.substack.com/p/why-ai-will-save-the-world
    documents = loader.load_data(file=Path(f'./data/why-ai-will-save-the-world.txt'), split_documents=True)
    # when first building the index
    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=create_service_context()
    )

    # persist index to disk, default "storage" folder
    index.storage_context.persist()

    return index


# 基于上下文信息分析
def context_data_query(input_text):
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="./storage")

    # loads index from storage
    index = load_index_from_storage(storage_context, service_context=create_service_context())

    # queries the index with the input text
    response = index.as_query_engine().query(input_text)

    return response.response


# 基于上下文与向量数据库中的回答分析
def data_querying(input_text):
    documents = loader.load_data(file=Path(f'./data/why-ai-will-save-the-world.txt'), split_documents=False)

    EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = INSTRUCTOR('hkunlp/instructor-large', device=EMBEDDING_DEVICE)
    embeddings = model.encode(input_text)
    tolist = embeddings.tolist()
    # reader = QdrantReader(host="140.179.185.67", grpc_port=6334, prefer_grpc=True)
    # documents = reader.load_data(collection_name="comments", query_vector=tolist, limit=5)
    client = QdrantClient(host="140.179.185.67", grpc_port=6334, prefer_grpc=True)
    elems = client.search(collection_name="comments", query_vector=tolist, limit=10)

    for elem in elems:
        text = elem.payload["f8"]
        vector = model.encode(text).tolist()
        doc = Document(doc_id=elem.id,
                       text=text,
                       # vector=elem.vector,
                       embedding=vector,
                       extra_info=elem.payload)
        documents.append(doc)

    index = ListIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    response = query_engine.query(input_text)
    return response.response


if __name__ == '__main__':
    index = data_ingestion_indexing()
    # iface = gr.Interface(fn=context_data_query,
    iface = gr.Interface(fn=data_querying,
                         inputs=gr.components.Textbox(lines=7, label="Enter your question"),
                         outputs="text",
                         title="Test")
    iface.launch(share=False)
# AI will not cause the commonly proposed risks, including mass unemployment and inequality.
