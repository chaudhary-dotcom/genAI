from langchain_huggingface import HuggingFacePipeline
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt
import os


st.header("Research summarizer")

@st.cache_resource
def load_llm():
    return HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        task="text-generation",
        pipeline_kwargs={
            # "max_new_tokens": 200,  
            "temperature": 0.5,
            "do_sample": False,
        }
    )
llm = load_llm()

# user_input = st.text_input("Enter your prompt")  # static prompt


# Dynamic Prompt 
paper_input = st.selectbox("Select Research Paper Name", ["Select...", "Attentation is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])

length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])

template = load_prompt("template.json")




if st.button("Summarize"):

    chain = template | llm

    result = chain.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
})
    
    st.write(result)



# if st.button("Summarize"):
#     # st.text("some random text")
#     if user_input:
#         result = llm.invoke(user_input)
#         st.write(result)

#     else:
#         st.warning("Please enter a prompt first.")


