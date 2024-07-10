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
        self.model = ChatOllama(model="llama2", temperature=0.01)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=25)
        self.prompt = PromptTemplate.from_template(
        """
        <s> [INST] Você é um assistente para tarefas de resposta a perguntas. Use SOMENTE as informações fornecidas no contexto 
        para responder à pergunta. Não adicione informações além do que está explicitamente declarado no contexto. 
        Se a informação não estiver presente no contexto, diga que você não pode responder com base nas informações fornecidas. 
        Responda sempre em português, traduzindo se necessário. Não crie respostas. [/INST] </s> 
        [INST] Pergunta: {question} 
        Contexto: {context} 
        Responda em português, com base apenas no contexto fornecido: [/INST]
        """
        )

    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = chromautils.filter_complex_metadata(chunks)

        vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        base_retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
            )
                # Adiciona um compressor de documento para refinar os resultados
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
