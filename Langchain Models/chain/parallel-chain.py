from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

model1 = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation"
)

model2 = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
    task="text-generation"
)

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",

    input_variables=["text"]

)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text \n {text}",

    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quize into a single document \n notes -> {notes} and {quiz}",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz": prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain


text = """
Artificial Intelligence (AI) is a branch of computer science that focuses on creating machines and software that can perform tasks that usually require human intelligence. These tasks include understanding language, recognizing images, solving problems, learning from data, and making decisions. AI is used in many areas of our daily lives, such as voice assistants, recommendation systems, online shopping, healthcare, banking, and self-driving technologies. Machine learning and deep learning are important parts of AI that allow computers to learn patterns from large amounts of data. As technology continues to improve, AI is becoming increasingly useful in education, business, science, and entertainment. However, AI also creates challenges related to privacy, security, employment, and responsible use. Therefore, understanding both the benefits and limitations of AI is important
"""
result = chain.invoke({"text": text})

print(result)

chain.get_graph().print_ascii()