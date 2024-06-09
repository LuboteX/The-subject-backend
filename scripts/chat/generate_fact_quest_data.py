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
import copy
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
from tqdm import tqdm
mypath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(mypath)
from utils.strer2 import encoder

class QuestOutput(BaseModel):
    key_facts: List[int] = Field(description="List of key_facts values you randomly choose")
    thought: str = Field(description="your thought and plan to generate the quest")
    quest: str = Field(description="Generated quest related to key_facts")

class GenerationData(BaseModel):
    generation: List[QuestOutput] = Field(description="List of quests and key_facts")

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
    
def create_model(modelName):
    if modelName == "custom":
        model = ChatOpenAI(model=chat_model, base_url=transit_url, openai_api_key=transit_key, temperature=0.5)
    elif modelName == "yi":
        model = ChatOpenAI(model='yi-medium-200k', base_url='https://api.lingyiwanwu.com/v1', openai_api_key=transit_key, temperature=0.75)
    else:
        model = None
    return model
def get_max_file_index(output_dir):
    # 获取目录下所有文件
    files = os.listdir(output_dir)
    max_index = 0
    pattern = re.compile(r'fact_quest_data_(\d{5})_(\d{2})\.json')
    
    for file in files:
        match = pattern.match(file)
        if match:
            index = int(match.group(1))
            if index > max_index:
                max_index = index
                
    return max_index

def quests_to_files(quests_outputs, templates, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取当前最高编号
    current_max_index = get_max_file_index(output_dir)
    # 处理并生成文件
    for i, item in enumerate(quests_outputs['generation']):
        templates[0]["input"]["quest"] = item['quest']
        templates[0]["output"] = item['key_facts']
        # 生成新的编号和文件名
        new_index = current_max_index + 1
        file_number = f"{new_index:05d}"
        loop_number = f"{i+1:02d}"
        output_path = os.path.join(output_dir, f"fact_quest_data_{file_number}_{loop_number}.json")
        
        # 保存文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(templates, f, ensure_ascii=False, indent=4)
if __name__ == '__main__':
    dataset_address = f'./document/training_data/fact_random_raw2_data'
    output_dir = f'./document/training_data/fact_random_encoded_data'
    train_type = "fact_search"
    # hate,affinity
    data_epoch = 500

    model = create_model("custom")
    prompts_address = f'./document/prompts/{train_type}.txt'
    if model is None:
        raise ValueError("Failed to create model. The model is None.")
    
    with open(prompts_address, encoding="utf-8") as f:
        doc = f.read()
    parser = JsonOutputParser(pydantic_object=GenerationData)
    prompt = PromptTemplate(
        template="\n{format_instructions}\n{query}\n{fact_dataset}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | model | parser
    json_files = [filename for filename in os.listdir(dataset_address) if filename.endswith('.json')]
    for filename in tqdm(json_files, desc="Processing files"):
        file_path = os.path.join(dataset_address, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            fact_dataset = json.load(file)
        filled_template = copy.deepcopy(fact_dataset)
        for item in filled_template:
            if 'output' in item:
                item['key_facts'] = item.pop('output')
            # Assuming `doc` is the query for chain.invoke
            response = chain.invoke({"query": doc,"fact_dataset": json.dumps(filled_template)})
            print("response:")
            print(response)
            quests_to_files(response,fact_dataset,output_dir)
