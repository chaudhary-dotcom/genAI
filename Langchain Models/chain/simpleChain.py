from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

template=PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=["topic"]
)

parser=StrOutputParser()

chain = template | model | parser

result = chain.invoke({"topic": "Bed Bug"})

print(result)

chain.get_graph().print_ascii()