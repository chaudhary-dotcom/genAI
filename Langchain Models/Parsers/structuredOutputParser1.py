from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="""
    Give exactly 3 facts about {topic}.

You MUST follow these formatting instructions:
{format_instructions}

Return ONLY the JSON object. Do not write explanations before or after the JSON.
    """,
    input_variables=['topic'],
    partial_variables={'format_instructions': parser.get_format_instructions()
    }

)



chain = template | model 



result = chain.invoke({'topic': 'black hole'})

# final_result = parser.parse(result.content)

print(result.content)