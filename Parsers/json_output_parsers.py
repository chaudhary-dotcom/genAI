import os
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template="""
    Give me the name, age and city of fictional person. {format_instructions}
    """,
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}

)

# prompt = template.format()

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

# Pipeline
chain = template | model | parser

result = chain.invoke({})
print(result)
