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

logging.basicConfig(level=logging.INFO)

class ChatPDF:
    def __init__(self, persist_directory="./vector_store"):
        self.model = ChatOllama(model="llama3.1", temperature=0)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1080, chunk_overlap=50)
        self.prompt = PromptTemplate.from_template(
        """
[INST] Você é um assistente especializado em pesquisar documentos. Siga estas instruções rigorosamente:

Responda apenas com informações presentes no contexto fornecido.
Se a informação não estiver no contexto, responda apenas "Não está no documento".
Se a palavra "quiz" for mencionada na pergunta, crie uma questão de múltipla escolha com 3 alternativas incorretas e 1 correta, baseada no contexto fornecido. Em seguida, indique a resposta correta.
Forneça respostas precisas e em português.

Pergunta: {question}
Contexto: {context}

Resposta:
[/INST]
        """
        )
        self.persist_directory = persist_directory
        self.vector_store = None
        self.retriever = None
        self.chain = None

        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # logging.info(f"Using device: {self.device}")

    def ingest(self, pdf_file_path: str):
        try:
            docs = PyPDFLoader(file_path=pdf_file_path).load()
            chunks = self.text_splitter.split_documents(docs)
            for i, chunk in enumerate(chunks):
                chunk.metadata = {
                    'source': 'course_module_name',
                    'chunk_id': i,
                    # Adicione outros metadados que desejar
                }          
            
            
            chunks = chromautils.filter_complex_metadata(chunks)
            
            

            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            self.vector_store = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=self.persist_directory)
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 3,
                    "score_threshold": 0.0001
                },
            )
            print(chunks)


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
