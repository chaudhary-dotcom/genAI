from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation"
)

prompt = PromptTemplate(
    template="write a joke about {topic}",
    input_variables=["topic"]
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template="summarize the joke {text}",
    input_variables=["text"]
)

chain = RunnableSequence(prompt, model, parser, prompt2, model, parser)

result = chain.invoke({"topic": "AI"})

print(result)