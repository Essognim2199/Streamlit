#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 15:01:06 2025

@author: richard
"""
import streamlit as st
import numpy
import getpass
import openai
import time
import os
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_ollama import ChatOllama
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import AIMessage, HumanMessage
import time

chemin = "/Streamlit/"

data = [] 
ld = PyPDFLoader(chemin+"Streamlit/"+"d5.pdf")
data = data + ld.load()

chunk_size = 1000
chunk_overlap = 200

rc_splitter = RecursiveCharacterTextSplitter(
chunk_size=chunk_size,
chunk_overlap=chunk_overlap,
separators=[".","\n", "\n\n","\n\n\n\n"," "])
docs = rc_splitter.split_documents(data)

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs= {'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

vectorstore = Chroma.from_documents(
docs,
embedding=embedding_function,
persist_directory=chemin
)

#vectorstore = Chroma(
#    persist_directory=chemin,
#    embedding_function=embedding_function
#)

retriever = vectorstore.as_retriever(
search_type="similarity",
search_kwargs={"k": 5}
)

prompt_t = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Tu es un assistant intelligent et précis. Réponds à la question suivante de l'utilisateur : {copy}, "
            "en te basant uniquement sur les informations fournies dans les consignes suivantes : {guidelines}. "
            "Ne fournis aucune information inventée, respecte strictement les consignes, et n'explique pas ta réponse et ne donne ni la source et ni la référence."
            "après avoir examiné les documents fournis, si tu ne trouve aucune information pertinente alors retourne comme réponse : Je suis désolé, je ne dispose pas d’informations à ce sujet."
            "Vas droit au but dans ta réponse n'ajoute rien.",
        ),
        ("human", "{copy}"),
    ]
)


#llm = init_chat_model("gpt-4o-mini", model_provider="openai")
llm = ChatNVIDIA(model="nvidia/llama-3.3-nemotron-super-49b-v1",api_key="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")

rag_chain = ({"guidelines": retriever, "copy": RunnablePassthrough()}| prompt_t|llm)

#----------------------------------------------------------------------------------------------------
st.set_page_config(layout="wide")
logo_path1 = "togo.png"
st.image(logo_path1, width=100)

st.title("Service Public TOGO")

st.markdown("""
    <style>
    /* Corps avec fond bleu très foncé */
    body {
        background: linear-gradient(to bottom, #195a9b, #d6eaff);  /* Bleu profond vers bleu pâle */
        color: #001a33;
        font-family: "Segoe UI", sans-serif;
    }

    .stApp {
        background: linear-gradient(to bottom, #195a9b, #d6eaff);
    }

    /* Conteneur principal avec texte noir */
    .css-18ni7ap, .css-1dp5vir {
        background-color: rgba(255, 255, 255, 0.96) !important;
        color: #000000 !important;
        border-radius: 12px;
        box-shadow: 0px 2px 12px rgba(0,0,0,0.1);
    }

    /* Champs texte */
    .stTextInput > div > div > input,
    .stTextArea textarea {
        background-color: #ffffff;
        color: #001a33;
        border: 1px solid #1a5fb4;
        border-radius: 5px;
    }

    /* Boutons bleu foncé */
    .stButton > button {
        background-color: #1a5fb4;
        color: #ffffff;
        border-radius: 8px;
        font-weight: bold;
        border: none;
    }

    .stButton > button:hover {
        background-color: #144a91;
        color: white;
    }

    /* Titres et séparateurs */
    hr {
        border-color: #1a5fb4;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #001a33;
    }

    a {
        color: #003399;
    }
    </style>
""", unsafe_allow_html=True)


logo_path = "lg.svg"
#st.image(logo_path)

st.sidebar.image(
    logo_path,
    width=400,
    caption="## OptimBrains TOGO"
)

st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            width: 300px;
        }
        [data-testid="stSidebar"] > div:first-child {
            width: 300px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        /* Modifier la largeur de la sidebar */
        section[data-testid="stSidebar"] {
            width: 450px !important;
        }

        /* Ajuster la zone de contenu principal en conséquence */
        div[data-testid="stSidebarContent"] {
            width: 450px !important;
        }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.markdown("""
---
## À propos de nous

OptimBrain est une entreprise togolaise fondée par de jeunes Togolais résidant au Togo, en France, au Canada et aux États-Unis. Son objectif est d’apporter une contribution significative au développement du pays et de la sous-région, particulièrement à une époque où l’intelligence artificielle représente l’avenir et un enjeu crucial de souveraineté nationale.

Spécialisée dans l’intelligence artificielle, l’optimisation et l’aide à la décision, OptimBrain s’engage à positionner le Togo et la sous-région ouest-africaine comme des acteurs incontournables dans ce domaine stratégique.

---
## Nos Services

- **Solutions IA sur mesure** 
- **Analyse de données** 
- **Optimisation de la production**
- **Planification des ressoucres**
- **Conseil en transformation digitale**
- **Formation**
""")

# --------Functions--------------------------------------------------
def stream_data(text, delay:float=0.01):
    """Streaming function"""
    if text is None:
        return  
    for word in text.split():
        yield word+ " "
        time.sleep(delay)

def generate_response(input_text):
    response = rag_chain.invoke(input_text)
    return response


# Chat history initialisation
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Bienvenue ! Je suis Esso, une intelligence artificielle conçue pour vous aider avec toutes vos questions concernant les services publics au Togo. Que puis-je faire pour vous aujourd’hui ?")]

# Print the chat history during the conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# Question from the user
user_query = st.chat_input("Ecrivez votre question ici...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))  # Add the question of user to the  history

    with st.chat_message("Human"):
        st.markdown(user_query)

    
    with st.chat_message("AI"):
        with st.spinner("Thinking ..."):
            response = generate_response(user_query)
            
            st.markdown(
                f"""
                <div style="
                    background-color: rgba(20, 20, 20, 0.7); 
                    padding: 1rem; 
                    border-radius: 10px; 
                    color: white; 
                    font-size: 16px;
                    line-height: 1.6;
                    margin-bottom: 1rem;
                ">
                    {response.content}
                </div>
                """,
                unsafe_allow_html=True
            )


      
    st.session_state.chat_history.append(AIMessage(content=response.content))   #Add the AI's answer to the  history
