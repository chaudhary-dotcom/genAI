from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser


url = "https://www.daraz.com.np/products/lux-soft-rose-body-wash-with-french-rose-almond-oil-245-ml-i454986640-s2011434712.html?scm=1007.51610.379274.0&pvid=0d9b0cc5-959b-4916-bcfe-11eafc2363db&search=flashsale&spm=a2a0e.tm80335409.FlashSale.d_454986640"

loader = WebBaseLoader(url)

docs = loader.load()

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation"
) 

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="Answer the following question \n {question} from the following text - \n {text}",
    input_variables=["question", "text"]
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"question": "what is the product we are tlking about ?", "text":  docs[0].page_content})



print(result)