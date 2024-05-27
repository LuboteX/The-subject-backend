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
import sys
import re
mypath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(mypath)
from utils.strer2 import encoder

config = configparser.ConfigParser()

# Read the config.ini file
config.read('config.ini')
file_template_path = "document/training_data/template.json"

# Access the values in the configuration file
persona_file = config['DEFAULT']['persona_file']
knowledge_file = config['DEFAULT']['knowledge_file']
transit_key = config['DEFAULT']['transit_key']
transit_url = config['DEFAULT']['transit_url']
memory_length = config['DEFAULT']['memory_length']
chat_model = config['DEFAULT']['chat_model']

def save_data(content, data_type):
    base_path = './document/training_data/raw_data/'
    base_filename = f'raw_data_{data_type}_'
    encode_filename = f'encode_data_{data_type}_'
    file_extension = '.json'
    file_template_path = "document/training_data/template.json"
    output_base_path = "utils/strer2/output/"
    
    # 获取现有文件列表并找出最大的编号
    existing_files = [f for f in os.listdir(base_path) if f.startswith(base_filename) and f.endswith(file_extension)]
    existing_numbers = [int(re.search(r'(\d{3})', f).group()) for f in existing_files if re.search(r'(\d{3})', f)]
    
    if existing_numbers:
        start_number = max(existing_numbers) + 1
    else:
        start_number = 1
        
    # 循环生成文件
    file_number = str(start_number).zfill(3)  # 生成三位数的编号，例如 001, 002
    output_filename = f'{encode_filename}{file_number}{file_extension}'
    filename = f'{base_filename}{file_number}{file_extension}'
    file_path = os.path.join(base_path, filename)
    
    with open(file_path, 'w', encoding='utf-8') as text_file:
        text_file.write(content)
        
    output_file_path = os.path.join(output_base_path, output_filename)
    
    encoder.encoder(file_path, file_template_path, output_file_path)
    
def create_model(modelName):
    if modelName == "custom":
        model = ChatOpenAI(model=chat_model, base_url=transit_url, openai_api_key=transit_key, temperature=0.8)
    else:
        model = None
    return model

if __name__ == '__main__':
    train_type = "affinity"
    # hate,affinity
    data_epoch = 10

    model = create_model("custom")
    prompts_address = f'./document/prompts/{train_type}.txt'
    if model is None:
        raise ValueError("Failed to create model. The model is None.")
    
    with open(prompts_address, encoding="utf-8") as f:
        doc = f.read()

    messages = [
        SystemMessage(content="根据以下描述生成一个json文件："),
        HumanMessage(content=doc),
    ]
    for i in range(data_epoch):
        print("data_generation:")
        print(i+1)
        response = model.invoke(messages)

        save_data(response.content, train_type)