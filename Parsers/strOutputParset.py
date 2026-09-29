import os
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# 1st prompt 
template1=PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=['topic']
)

template2=PromptTemplate(
    template="write a 5 line summary on the following text. /n {text}",
    input_variables=["text"]
)

# pipeline 
parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'black hole'})
print(result)