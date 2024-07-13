from collections import deque
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from load_model_for_inference import ov_model, tokenizer


model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model_kwargs = {'device': 'cpu'}


embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

DB_FAISS_PATH = 'vectorstores/db_faiss'

custom_prompt_template = """[INST] <<SYS>>
 Use the following pieces of information to answer the user's question. 

Context:
{context}

Question:
{question}

Return the helpful answer below.
[/INST]

Helpful answer:

"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector store
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def create_llm(model, tokenizer):
    # Create the Hugging Face text generation pipeline
    text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    max_new_tokens=350,
    tokenizer=tokenizer, # Adjust as needed for desired length
    #do_sample=True,   # Enable sampling for more natural text
    #temperature=1   Control randomness (higher = more creative)
    #top_p=0.95,       # Nucleus sampling for better quality
    #no_repeat_ngram_size=2, # Prevent repetition of phrases
    #early_stopping=True, # Stop when model thinks it's done
    pad_token_id=tokenizer.eos_token_id # Replace pad tokens with EOS
    )

    # Wrap the pipeline in LangChain's HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return llm

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt},
    )
    return qa_chain

def qa_bot(model, tokenizer):
    embeddings = HuggingFaceEmbeddings(model_name=model_name,model_kwargs=model_kwargs)
    db = FAISS.load_local(DB_FAISS_PATH, embeddings,allow_dangerous_deserialization=True)
    llm = create_llm(model, tokenizer)
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa



def final_result(query, model, tokenizer):
    qa = qa_bot(model, tokenizer)

    qa_result = qa({'query': query})

    response = qa_result['result']


    helpful_answer = response.split('Helpful answer:')[-1].strip()

    return helpful_answer.strip()


