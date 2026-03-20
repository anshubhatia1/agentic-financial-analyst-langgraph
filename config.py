import os
from dotenv import load_dotenv
import faiss as faiss_lib
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS as FAISSClass
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from openai import OpenAI
from tavily import TavilyClient

load_dotenv()

## LLM ##
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.2,
    max_retries=4,
)

# ## Embeddings ##
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-mpnet-base-v2",
#     model_kwargs={"device": "cpu"},
#     encode_kwargs={"normalize_embeddings": False},
# )

# ## FAISS vector store ##
# _dim        = len(embeddings.embed_query("hello world"))
# _index      = faiss_lib.IndexFlatL2(_dim)
# faiss_store = FAISSClass(
#     embedding_function=embeddings,
#     index=_index,
#     docstore=InMemoryDocstore(),
#     index_to_docstore_id={},
# )