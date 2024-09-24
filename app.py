import tiktoken
import os
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams
from typing import List
from chainlit.types import AskFileResponse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_openai.llms import OpenAI
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain.chains.conversation.base import ConversationChain
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import (ChatMessagePromptTemplate, SystemMessagePromptTemplate, 
                                    AIMessagePromptTemplate, HumanMessagePromptTemplate)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
import chainlit as cl

from dotenv import load_dotenv; _ = load_dotenv()

RAG_PROMPT = """
Please answer the question below using the provided context. Be as detailed as you can be based on the contextual information. 
If the question cannnot be answered using the context, politely state that you can't answer that question.

Question:
{question}

Context:
{context}
"""

def get_rag_chain():
    """Fetches a simple RAG chain"""
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    embedding = HuggingFaceEmbeddings(
        model_name="deman539/nomic-embed-text-v1",
        model_kwargs={'trust_remote_code': True}
    )
    retriever = QdrantVectorStore.from_existing_collection(
        collection_name='ai_ethics_nomicv1_finetuned',
        embedding=embedding,
        url=os.environ.get('QDRANT_DB'),
        api_key=os.environ.get('QDRANT_API_KEY')
    ).as_retriever()
    llm = ChatOpenAI(model='gpt-4o', temperature=0)
    rag_chain = ({'context': retriever, 'question': RunnablePassthrough()}
             | prompt
             | llm)
    return rag_chain


@cl.on_chat_start
async def on_chat_start():
    """Initialization of the application"""
    msg = cl.Message(
        content="", disable_human_feedback=True
    )
    await msg.send()
    chain = get_rag_chain()
    # Let the user know that the system is ready
    msg.content = """
    I'm ready to answer any of your questions about the framework for [AI Bill of Rights](https://www.whitehouse.gov/wp-content/uploads/2022/10/Blueprint-for-an-AI-Bill-of-Rights.pdf)
    and [NIST AI Risk Management Framework](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf)
    Ask away!
    """
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    """Run on every user message"""
    chain = cl.user_session.get("chain")

    msg = cl.Message(content="")
    async for resp in chain.astream(message.content):
        await msg.stream_token(resp.content)

    await msg.send()
