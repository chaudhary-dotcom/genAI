from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)


class Person(BaseModel):
    name: str = Field(description='Name of the person')
    age: int =Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template=""" 
Generate the name, age  and city of the {place} person \n {format_instructions}
""", 
    input_variables=['place'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
    

)

# prompt = template.invoke({'place': 'nepali'})

# result = model.invoke(prompt)

# print(result.content)

# pipeline 
chain = template | model
result = chain.invoke({'place': 'british'})
print(result.content)