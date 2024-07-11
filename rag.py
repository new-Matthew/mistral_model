from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores import utils as chromautils

class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self, temperature=0.02, top_p=0.05, top_k=1):
        self.model = ChatOllama(
            model="llama3:8b",
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            You are an assistant for question-answering tasks. Use only the provided context to answer the question.
            If the answer is not in the context, respond with "I don't know." Provide a concise answer.
            Question: {question}
            Context: {context}
            Answer:
            """
        )

    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = chromautils.filter_complex_metadata(chunks)

        vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.7,
            },
        )

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."

        response = self.chain.invoke(query)
        
        if "I don't know" in response:
            return "The information you are looking for is not available in the provided PDF."
        return response

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
