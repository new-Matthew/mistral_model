from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores import utils as chromautils
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.embeddings import HuggingFaceEmbeddings
#from langchain_community.embeddings import FastEmbedEmbeddings
class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model="llama3:8b", temperature=0.1)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=300)
        self.prompt = PromptTemplate.from_template(
        """
        <s> [INST] Você é um assistente de resposta a perguntas. Use SOMENTE as informações fornecidas no contexto 
        para responder à pergunta. Responda o que está explicitamente declarado no contexto. 
        Se a informação não estiver presente no contexto, diga que você não pode responder com base nas informações fornecidas. 
        Responda sempre em português, traduzindo se necessário.[/INST] </s> 
        [INST] Pergunta: {question} 
        Contexto: {context} 
        Responda em português: [/INST]
        """
        )

    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = chromautils.filter_complex_metadata(chunks)

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        self.vector_store = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory="./vector_store")
        base_retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 4,
                "score_threshold": 0.6,
            },
        )

        compressor = LLMChainExtractor.from_llm(self.model)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )


        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())



    def ask(self, query: str):
        if not self.chain:
            return "Por favor, adicione um documento em PDF primeiro."

        return self.chain.invoke(query)
        
    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
