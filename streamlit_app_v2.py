import streamlit as st
import openai
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from llama_index.core import Settings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

# Load the stored environment variables
load_dotenv()

st.set_page_config(page_title="Chat with the Quads/Badfish docs", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = os.environ["OPENAI_API_KEY"]
st.title("Quads and Badfish chat bot 🛺🐡💬")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me any question about Quads or Badfish!"}
    ]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the QUADS and Badfish docs – hang tight! This should take 1-2 minutes."):
        loader = WebBaseLoader(["https://raw.githubusercontent.com/redhat-performance/badfish/master/README.md","https://raw.githubusercontent.com/redhat-performance/quads/master/README.md"])
        data = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
        docs = splitter.split_documents(data)
        
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
            temperature=0.1,
            repetition_penalty=1.03,
        )
        Settings.llm = llm
        Settings.embed_model = lc_embed_model

        vectorstore = Chroma.from_documents(documents=docs, embedding=lc_embed_model)

        memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )

        retrieval_qa = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
        )
        return retrieval_qa

retrieval_qa = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        # RetrievalQA
        st.session_state.chat_engine = retrieval_qa

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine({"question": prompt})
            st.write(response["answer"])
            message = {"role": "assistant", "content": response["answer"]}
            st.session_state.messages.append(message) # Add response to message history
