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
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
import re
from typing import List, Dict
from langchain.output_parsers.openai_tools import JsonOutputToolsParser
mypath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(mypath)
from utils.strer2 import encoder

class Generation1(BaseModel):
    plot: str = Field(description="plot summary of generation 1")
    chat_history: List[str] = Field(description="chat history between characters in generation 1")
    facts: List[str] = Field(description="list of facts about generation 1")

class Generation2(BaseModel):
    plot: str = Field(description="plot summary of generation 2")
    chat_history: List[str] = Field(description="chat history between characters in generation 2")
    facts: List[str] = Field(description="list of facts about generation 2")

class Generation3(BaseModel):
    plot: str = Field(description="plot summary of generation 3")
    chat_history: List[str] = Field(description="chat history between characters in generation 3")
    facts: List[str] = Field(description="list of facts about generation 3")

class Generations(BaseModel):
    generation1: Generation1 = Field(description="generation 1 details")
    generation2: Generation2 = Field(description="generation 2 details")
    generation3: Generation3 = Field(description="generation 3 details")

config = configparser.ConfigParser()

# Read the config.ini file
config.read('config.ini')
file_template_path = "document/training_data/fact_template.json"

# Access the values in the configuration file
persona_file = config['DEFAULT']['persona_file']
knowledge_file = config['DEFAULT']['knowledge_file']
transit_key = config['DEFAULT']['transit_key']
transit_url = config['DEFAULT']['transit_url']
memory_length = config['DEFAULT']['memory_length']
chat_model = config['DEFAULT']['chat_model']

def save_data(content, data_type):
    base_path = './document/training_data/fact_raw_data/'
    base_filename = f'fact_raw_data_{data_type}_'
    encode_filename = f'fact_encode_data_{data_type}_'
    file_extension = '.json'
    file_template_path = "document/template/fact_template.json"
    output_base_path = "./document/training_data/fact_encoded_data/"
    
    # 获取现有文件列表并找出最大的编号
    existing_files = [f for f in os.listdir(base_path) if f.startswith(base_filename) and f.endswith(file_extension)]
    existing_numbers = [int(re.search(r'(\d{5})', f).group()) for f in existing_files if re.search(r'(\d{5})', f)]
    
    if existing_numbers:
        start_number = max(existing_numbers) + 1
    else:
        start_number = 1
        
    # 循环生成文件
    file_number = str(start_number).zfill(5)  # 生成三位数的编号，例如 00001, 00002
    output_filename = f'{encode_filename}{file_number}{file_extension}'
    filename = f'{base_filename}{file_number}{file_extension}'
    file_path = os.path.join(base_path, filename)
    if isinstance(content, dict):
        # 如果内容是字典，则保存为 JSON 格式
        with open(file_path, 'w', encoding='utf-8') as text_file:
            json.dump(content, text_file, ensure_ascii=False, indent=4)
        print("内容已成功保存为 JSON 文件")
    elif isinstance(content, str):
        # 如果内容是字符串，则直接写入文件
        with open(file_path, 'w', encoding='utf-8') as text_file:
            text_file.write(content)
        print("内容已成功保存为文本文件")
    else:
        print("内容类型不支持")
        
    output_file_path = os.path.join(output_base_path, output_filename)
    
    encoder.encoder(file_path, file_template_path, output_file_path)
    
def create_model(modelName):
    if modelName == "custom":
        model = ChatOpenAI(model=chat_model, base_url=transit_url, openai_api_key=transit_key, temperature=0.7)
    elif modelName == "yi":
        model = ChatOpenAI(model='yi-medium-200k', base_url='https://api.lingyiwanwu.com/v1', openai_api_key=transit_key, temperature=0.75)
    else:
        model = None
    return model

if __name__ == '__main__':
    train_type = "fact1"
    # hate,affinity
    data_epoch = 500

    model = create_model("custom")
    prompts_address = f'./document/prompts/{train_type}.txt'
    if model is None:
        raise ValueError("Failed to create model. The model is None.")
    
    with open(prompts_address, encoding="utf-8") as f:
        doc = f.read()
    # message = [
    #     SystemMessage(content="你是一位精通中文的编剧，根据以下描述生成一个json文件："),
    #     HumanMessage(content=doc),
    # ]
    parser = JsonOutputParser(pydantic_object=Generations)
    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | model | parser

    for i in range(data_epoch):
        print("data_generation:")
        print(i+1)
        response = chain.invoke({"query": doc})
        save_data(response, train_type)