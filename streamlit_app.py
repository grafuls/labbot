import streamlit as st
import openai
import os
# from llama_index.llms.openai import OpenAI
try:
  from llama_index import VectorStoreIndex, ServiceContext
except ImportError:
  from llama_index.core import VectorStoreIndex, ServiceContext, Document
from langchain_community.document_loaders import WebBaseLoader
# from llama_index.core.tools import QueryEngineTool
# from llama_index.core.query_engine import RouterQueryEngine
# from llama_index.core.selectors import (
    # PydanticSingleSelector,
# )
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
from llama_index.core import Settings

st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
st.title("Quads and Badfish docs chat bot 🐡💬🦙")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Quads or Badfish open-source Python library!"}
    ]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the QUADS and Badfish docs – hang tight! This should take 1-2 minutes."):
        loader = WebBaseLoader(["https://raw.githubusercontent.com/redhat-performance/badfish/master/README.md","https://raw.githubusercontent.com/redhat-performance/quads/master/README.md"])
        data = loader.load()
        # for doc in data:
        #     doc = Document(doc)
        splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
        docs = splitter.split_documents(data)
        # import ipdb;ipdb.set_trace()
        
        from langchain_community.embeddings import HuggingFaceEmbeddings

        lc_embed_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        from langchain_community.llms import HuggingFaceEndpoint
        server_url = "https://llama-2-7b-chat-perfconf-hackathon.apps.dripberg-dgx2.rdu3.labs.perfscale.redhat.com"
        llm = HuggingFaceEndpoint(
            endpoint_url=server_url,
            max_new_tokens=256,
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
            temperature=0.5,
            repetition_penalty=1.03,
            # streaming=True,
        )
        Settings.llm = llm
        Settings.embed_model = lc_embed_model

        # service_context = ServiceContext.from_defaults(
            # llm=llm, 
            # embed_model=lc_embed_model
        # )
        from langchain_community.vectorstores import Chroma
        vectorstore = Chroma.from_documents(documents=docs, embedding=lc_embed_model)

        # RAG prompt
        from langchain import hub
        i_prompt = hub.pull("grafuls/rag-prompt")
        
        # index = VectorStoreIndex.from_documents(docs)
        return llm, vectorstore, i_prompt

llm, vectorstore, i_prompt = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        # RetrievalQA
        from langchain.chains import RetrievalQA
        st.session_state.chat_engine = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": i_prompt}
        )

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine({"query": prompt})
            st.write(response["result"])
            message = {"role": "assistant", "content": response["result"]}
            st.session_state.messages.append(message) # Add response to message history
