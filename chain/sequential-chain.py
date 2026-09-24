from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1=PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)


template2=PromptTemplate(
    template="Generate a 5 pointer summary from the following text \n {text}",
    input_variables=["text"]
)

parser=StrOutputParser()


chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "Mount Everest"})

print(result)

chain.get_graph().print_ascii()