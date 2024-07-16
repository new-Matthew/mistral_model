import logging
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores import utils as chromautils
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import LLMChainExtractor
# import torch
logging.basicConfig(level=logging.INFO)

class ChatPDF:
    def __init__(self, model_name="llama3", temperature=0.1, chunk_size=250, chunk_overlap=50,
                 embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", persist_directory="./vector_store",
                 k=3, score_threshold=0.2):
        self.model = ChatOllama(model=model_name, temperature=temperature)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.prompt = PromptTemplate.from_template(
        """
        <s>[INST]
        Você é um assistente técnico especializado em pesquisar documentos. responda somente o que está no contexto. Se a informação não estiver no contexto, diga "Não está no documento"
        [/INST] </s>
        Pergunta: {question}
        Contexto: {context}
        Resposta precisa em português
        """
        )
        self.embedding_model_name = embedding_model_name
        self.persist_directory = persist_directory
        self.k = k
        self.score_threshold = score_threshold
        self.vector_store = None
        self.retriever = None
        self.chain = None

        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # logging.info(f"Using device: {self.device}")

    def ingest(self, pdf_file_path: str):
        try:
            docs = PyPDFLoader(file_path=pdf_file_path).load()
            chunks = self.text_splitter.split_documents(docs)
            chunks = chromautils.filter_complex_metadata(chunks)


            embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

            self.vector_store = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=self.persist_directory)
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self.k,
                    "score_threshold": self.score_threshold
                },
            )
            print(chunks)

            # compressor = LLMChainExtractor.from_llm(self.model)
            # self.retriever = ContextualCompressionRetriever(
            #     base_compressor=compressor,
            #     base_retriever=base_retriever
            # )

            self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                          | self.prompt
                          | self.model
                          | StrOutputParser())

            logging.info("Document ingestion and setup completed successfully.")
        except Exception as e:
            logging.error(f"Error during document ingestion: {e}")

    def ask(self, query: str):
        if not self.chain:
            logging.warning("No document loaded. Please ingest a PDF document first.")
            return "Por favor, adicione um documento em PDF primeiro."

        try:
            return self.chain.invoke(query)
        except Exception as e:
            logging.error(f"Error during query processing: {e}")
            return "Ocorreu um erro ao processar sua consulta."

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
        logging.info("Cleared all loaded documents and configurations.")
