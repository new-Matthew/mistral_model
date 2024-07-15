from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores import utils as chromautils
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model="llama3", temperature=0.1)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.prompt = PromptTemplate.from_template(
        """
        <s> [INST] Você é um assistente especializado em responder perguntas baseadas estritamente no conteúdo do documento fornecido. 
        Use APENAS as informações presentes no contexto abaixo. Não adicione informações externas ou suposições.
        Se a informação não estiver explicitamente no contexto, responda: "Não posso responder a essa pergunta com base nas informações fornecidas no documento.". [/INST] </s> 
        [INST] Pergunta: {question}
        Contexto: {context}
        Responda em português, seja preciso e conciso: [/INST]
        """
        )
        
    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = chromautils.filter_complex_metadata(chunks)
        
        vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings(), persist_directory="./vector_store")
        base_retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 5,
                "score_threshold": 0.5,
                }
                )
                
        compressor = LLMChainExtractor.from_llm(self.model)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        print(base_retriever)
        print(compressor)
        print(chunks)

        
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
