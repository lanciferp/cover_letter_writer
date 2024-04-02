from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import DirectoryLoader

from langchain import hub
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate

from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import yaml
import argparse


parser = argparse.ArgumentParser(
    prog="Cover Letter Writer",
    description="Writes a cover letter for you based on a corpus of resume's and cover letters you create."
)
parser.add_argument('config', type=str)

args = parser.parse_args()

print(os.getcwd())
yaml_path = os.path.join(".", "configs", args.config)
with open(yaml_path, 'r') as file:
    config = yaml.safe_load(file)

os.environ["OPENAI_API_KEY"] = config["openai_key"]

loader = DirectoryLoader(config["knowledge_base_path"], loader_cls=Docx2txtLoader)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})
llm = ChatOpenAI(model_name="gpt-4", temperature=0.9)
corrector_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


cover_letter_template = PromptTemplate(
    input_variables=['context', 'description'],
    template='You are an assistant for writing cover letters. Write up to 5 paragraphs describing why I would be a '
             'good fit for the position. Do not make up expertise or experience that I do not have listed on my resume.'
             'Do tell stories you find written in the example cover letters if they apply.'
             'Write me a cover letter for the following position. '
             '\nContext:{context}'
             '\nDescription: {description}'
)

corrector_template = PromptTemplate(
    input_variables=['context', 'cover_letter'],
    template='You are an assistant who corrects cover letters for accuracy. Find any mistakes in this cover letter,'
             ' and correct them. If there are no mistake, say there are no mistakes.'
             '\nContext:{context}'
             '\nCover Letter: {cover_letter}'
)

refiner_template = PromptTemplate(
    input_variables=['cover_letter', 'corrections'],
    template='You are an assistant who fixed cover letters. Here is a cover letter and a set of corrections.'
             'Fix the cover letter, and output a final version. It should be 5 paragraphs long.'
             '\nCover Letter: {cover_letter}'
             '\nCorrections: {corrections}'
)

initial_chain = (
        {"context": retriever | format_docs, "description": RunnablePassthrough()}
        | cover_letter_template
        | llm
        | StrOutputParser()
)

corrector_chain = (
        {"context": retriever | format_docs, "cover_letter": RunnablePassthrough()}
        | corrector_template
        | corrector_llm
        | StrOutputParser()
)

refinement_chain = (
        {"cover_letter": RunnablePassthrough(), "corrections": RunnablePassthrough()}
        | refiner_template
        | corrector_llm
        | StrOutputParser()
)

# app framework
st.title("Cover Letter Writer")
description = st.text_area("Paste the Job Description Here!")

if description:
    response = initial_chain.invoke(description)
    st.markdown(response)
    correction = corrector_chain.invoke(response)
    st.markdown(correction)
    if "no mistakes" not in correction:
        refinement = refinement_chain.invoke([response, correction])
        st.markdown(refinement)

vectorstore.delete_collection()
