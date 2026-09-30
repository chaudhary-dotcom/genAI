from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnableParallel, RunnablePassthrough

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation"
)

def word_count(text):
    return len(text.split())


prompt = PromptTemplate(
    template="Summarize the text in 50 words - \n {topic}",
    input_variables=["topic"]
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

loader = TextLoader("balendra.txt", encoding="utf-8")

docs = loader.load()

chain = prompt | model | parser

parallel_chain = RunnableParallel(
    {
        "topic": RunnablePassthrough(),
        "word_count": RunnableLambda(word_count)
    }
)

final_chain = chain | parallel_chain


result = final_chain.invoke({"topic": docs[0].page_content})

print(result)

