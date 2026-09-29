from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda


llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        # "max_new_tokens": 30,
        "do_sample": False,
        "return_full_text": False, 
    }
)

model=ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template= "Classify the sentiment of the following text into 'positive' or 'negative' \n {feedback}",
    input_variables=["feedback"]
)


classifier_chain = prompt1 | model | parser | RunnableLambda(lambda text: "positive" if "positive" in text.lower() else "negative")

prompt2 = PromptTemplate(
    template = "Write an appropriate response to this positive feedback \n {feedback}",
    input_variables = ["feedback"]
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback \n {feedback}",
    input_variables=["feedback"]
)

pipeline_input = RunnableParallel({
    "feedback": RunnableLambda(lambda x: x["feedback"]),
    "sentiment": classifier_chain
})

branch_chain = RunnableBranch(
    (lambda x: x["sentiment"] == "positive", prompt2 | model | parser),
    (lambda x: x["sentiment"] == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find the sentiment")

)

chain = pipeline_input | branch_chain

print(chain.invoke({"feedback": "Product is amazing"}))