from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 30,
        "do_sample": False,
        "return_full_text": False,  # Prevents echoing the prompt and chat tokens
    }
)

model = ChatHuggingFace(llm=llm)

class Feedback(BaseModel):
    # Added "neutral" because 0.5B models occasionally classify borderline inputs as neutral
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Sentiment of the feedback"
    )

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sentiment analyzer. Always respond with a single JSON object in the format: {{\"sentiment\": \"positive\"}} or {{\"sentiment\": \"negative\"}}."),
    ("human", "Great battery life and clear screen."),
    ("ai", '{{"sentiment": "positive"}}'),
    ("human", "Broke after two days, completely useless."),
    ("ai", '{{"sentiment": "negative"}}'),
    ("human", "{feedback}")
])

classifier_chain = prompt | model | parser2

print(classifier_chain.invoke({"feedback": "This is best smartphone"}))