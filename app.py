import json
import os
import sys
import boto3
import streamlit as st

## Using Titan Embeddings to generate embeddings for the data
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader

## Vector Embedding and Vector Store
from langchain.vectorstores import FAISS

## LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Bedrock Client
bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
bedrock_embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v2:0', model_kwargs={'input_type': 'search_document'}, client=bedrock)

def data_ingestion():
    # Load the data
    loader = PyPDFDirectoryLoader('data/')
    documents = loader.load()

    # Split the data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    return texts

def vector_embedding(docs):
    vector_store = FAISS.from_documents(docs, bedrock_embeddings)
    vector_store.save_local('faiss_index')

def get_claude_llm():
    return bedrock.create_llm(model_id='anthropic.claude-3-5-sonnet-20240620-v1:0',
                              model_kwargs={'input_type': 'search_document'}, client=bedrock)
def get_llama_llm():
    return bedrock.create_llm(model_id='meta.llama3-70b-instruct-v1:0',
                             model_kwargs={'input_type': 'search_document', 'max_tokens': 1000}, client=bedrock)

## Prompt Template
prompt_template = """
You are a helpful assistant that can answer questions about the following text:
{context}

Question: {question}
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']

def main():
    st.set_page_config("Chat PDF")
    
    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                vector_embedding(docs)
                st.success("Done")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
            llm=get_claude_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
            llm=get_llama_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")


if __name__ == "__main__":
    main()



    













