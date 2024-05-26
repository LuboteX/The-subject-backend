from operator import itemgetter
import os
from langchain_community.chat_models import BedrockChat
from langchain_community.vectorstores.chroma import Chroma
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from langchain_core.messages import BaseMessage
from langchain_community.utils.math import cosine_similarity
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
import os
import configparser
import json

config = configparser.ConfigParser()

# Read the config.ini file
config.read('config.ini')

# Access the values in the configuration file
persona_file = config['DEFAULT']['persona_file']
knowledge_file = config['DEFAULT']['knowledge_file']
transit_key = config['DEFAULT']['transit_key']
transit_url = config['DEFAULT']['transit_url']
memory_length = config['DEFAULT']['memory_length']
chat_model = config['DEFAULT']['chat_model']

prompts_address = "./document/prompts/template2.txt"

def create_model(modelName):
    if modelName == "custom":
        model = ChatOpenAI(model=chat_model, base_url=transit_url, openai_api_key=transit_key, temperature=0.8)
    else:
        model = None
    return model

if __name__ == '__main__':
    model = create_model("custom")

    if model is None:
        raise ValueError("Failed to create model. The model is None.")
    
    with open(prompts_address, encoding="utf-8") as f:
        doc = f.read()

    messages = [
        SystemMessage(content="根据以下描述生成一个json文件："),
        HumanMessage(content=doc),
    ]

    response = model.invoke(messages)
    
    # Save response content to a text file
    txt_file = 'response_content.txt'

    with open(txt_file, 'w', encoding='utf-8') as text_file:
        text_file.write(response.content)
    
    print(response.content)
