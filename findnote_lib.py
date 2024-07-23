
from langchain.chains import RetrievalQA
import os
import getpass
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import keys as key
import google.generativeai as genai
import google.generativeai as palm

palm.configure(api_key=key.api_key)

models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
model = models[0].name
print(model)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=key.api_key)

api_key = key.api_key


def get_index():
    loader = TextLoader("content.txt", encoding = 'UTF-8')
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n","\n\n","."]
)


    docs = text_splitter.split_documents(document)


    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=docs,
                                        embedding=embeddings)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold = 0.7)
    return retriever
    


def get_llm():
    llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key= key.api_key)

    return llm

def get_memory(): #create memory for this chat session
    
    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True) #Maintains a history of previous messages
    
    return memory

def get_rag_chat_response(input_text, memory, index): #chat client function
    
    llm = get_llm()
    
    conversation_with_retrieval = ConversationalRetrievalChain.from_llm(llm, index.vectorstore.as_retriever(), memory=memory, verbose=True)
    
    chat_response = conversation_with_retrieval.invoke({"question": input_text}) #pass the user message and summary to the model
    
    return chat_response['answer']




# prompt_template = """Given the following context and a question, generate an answer based on this context only.
# In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
# If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer. The answer should not be the exact as per the source document.
# It can be a summary as well.

# CONTEXT: {context}

# QUESTION: {question}"""


# PROMPT = PromptTemplate(
#     template=prompt_template, input_variables=["context", "question"]
# )
# chain_type_kwargs = {"prompt": PROMPT}



# chain = RetrievalQA.from_chain_type(llm=llm,
#                             chain_type="stuff",
#                             retriever=retriever,
#                             input_key="query",
#                             return_source_documents=True,
#                             chain_type_kwargs=chain_type_kwargs)

