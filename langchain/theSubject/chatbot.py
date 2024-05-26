from operator import itemgetter
import os
from langchain_community.chat_models import BedrockChat
from langchain_community.vectorstores.chroma import Chroma
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough,RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter 
import tiktoken
from langchain_core.messages import BaseMessage
from langchain_community.utils.math import cosine_similarity
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

import os

persona_file = './langchain/theSubject/persona.txt'
knowledge_file = './langchain/theSubject/knowledge.txt'
memory_length = 6
# model = ChatOpenAI()
memory = ConversationBufferWindowMemory(k=memory_length, return_messages=True)
long_term_memory = ConversationBufferMemory(return_messages=True)
embeddings = OpenAIEmbeddings(model='text-embedding-3-large',base_url="https://api.aigcbest.top/v1",openai_api_key = qdd_key)

def create_model(modelName,key = None):
    if modelName == "haiku":
        model = BedrockChat(  # 使用AWS 的anthropic
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        streaming=True,
        model_kwargs={"temperature": 0.6},
        region_name="us-east-1")
    elif modelName == "gpt4":
        model = ChatOpenAI(model="gpt-4-0125-preview",openai_api_key = openai_key, temperature=0.5)
    elif modelName == "gpt3.5":
        model = ChatOpenAI(model="gpt-3.5-turbo-0125",openai_api_key = openai_key, temperature=0.5)
    elif modelName == "custom":
        model = ChatOpenAI(model="claude-3-haiku-20240307",base_url="https://api.aigcbest.top/v1",openai_api_key = qdd_key, temperature=0.5)
    else:
        model = None
    return model
    

def sort_indexes(arr):
    indexed_arr = list(enumerate(arr))
    
    sorted_indexed_arr = sorted(indexed_arr, key=lambda x: x[1], reverse=True)
    
    sorted_indexes = [x[0] for x in sorted_indexed_arr]
    
    return sorted_indexes

def num_tokens_from_string(string: str, encoding_name: str = 'cl100k_base') -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_long_term_memory(chat_history: list[BaseMessage]):
    query = ''
    for i, mes in enumerate(chat_history):
        query += f"Player: {mes.content}\n\n" if i % 2 == 0 else f"Bot: {mes.content}"
    query_embedding = embeddings.embed_query(query)
    dialogues = []
    temp = ''
    long_term_history = long_term_memory.load_memory_variables({})['history']
    if len(long_term_history) == 0:
        return {'query': query, 'chat_history': chat_history}
    for i, mes in enumerate(long_term_history):
        temp += f"Player: {mes.content}\n\n" if i % 2 == 0 else f"Bot: {mes.content}"
        if i % 2 == 1:
            dialogues.append(temp)
            temp = ''
    dialogues_emb = embeddings.embed_documents(dialogues)
    similarity = cosine_similarity([query_embedding], dialogues_emb)[0]
    most_similar = sort_indexes(similarity)
    num_tokens = 0
    selected_indexes = []
    for i in most_similar:
        num_tokens += num_tokens_from_string(dialogues[i])
        query += dialogues[i]
        if num_tokens > 4096 or similarity[i] < 0.6:
            break
        selected_indexes.append(i)
    selected_indexes = sorted(selected_indexes, reverse=True)
    for i in selected_indexes:
        chat_history.insert(0, long_term_history[i * 2 + 1])
        chat_history.insert(0, long_term_history[i * 2])
    return {'query': query, 'chat_history': chat_history}

def get_short_term_memory(input: dict) -> list[BaseMessage]:
    messages = memory.load_memory_variables({})['history']
    messages.append(HumanMessage(input["query"]))
    return messages


def get_prompt(chat_history: dict) -> dict:
    query = chat_history['query']
    chat_history = chat_history['chat_history']
    with open(persona_file, encoding="utf-8") as f:
        persona = f.read()
    world_knowledge = retriever.invoke(query)
    return {"memory": chat_history, "rules": "This is a roleplay", "persona": persona, "world_knowledge": world_knowledge}



if __name__ == '__main__':
    model = create_model("custom")

    if model is None:
        raise ValueError("Failed to create model. The model is None.")
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{rules}\n{persona}\n{world_knowledge}\n[Start Roleplay]"),
            MessagesPlaceholder(variable_name="memory"),
        ]
    )
    
    with open(knowledge_file, encoding="utf-8") as f:
        doc = f.read()
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
    docs = text_splitter.create_documents(texts=[doc])
    all_splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    output_parser = StrOutputParser()
    while True:
        question = input("输入：")
        chain = (
            {"query": RunnablePassthrough()}
            | RunnableLambda(get_short_term_memory)
            | RunnableLambda(get_long_term_memory)
            | RunnableLambda(get_prompt)
            | prompt
            | model
        )

        response = chain.invoke(question)
        temp_memory = memory.load_memory_variables({})['history']
        if (len(temp_memory) == 2*memory_length):
            long_term_memory.save_context({'input': temp_memory[0].content}, {'output': temp_memory[1].content})
        memory.save_context({'input': question}, {'output': response.content})

        print(response.content)