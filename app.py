import streamlit as st
import os
import tempfile
import logging
import requests
from PIL import Image
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader,
    UnstructuredHTMLLoader, UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

# ✅ Secure API Key Handling
API_KEY = "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"  # Replace with your Groq API Key
OCR_API_KEY = "K86466961488957"  # Replace with your OCR.Space API Key
os.environ["GROQ_API_KEY"] = API_KEY

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Define Math Chatbot System Prompt
SYSTEM_PROMPT = """
System Prompt: You are a highly skilled mathematician specializing in advanced concepts such as differential geometry, topology, and abstract algebra.
You apply these concepts to real-world problems, particularly in physics and computer science. Your explanations are clear, rigorous, and structured.

Instructions for Generating Responses:
1. Extract data from the uploaded file and present it in a structured format with clear steps.Ensure that mathematical expressions are properly formatted and readable.
2. Use a systematic, step-by-step approach like a professor explaining concepts.
3. Break down problems into smaller logical steps before proceeding to calculations.
4. Use proper LaTeX formatting for mathematical expressions.
5. Provide detailed reasoning behind each step to ensure clarity.
6. If multiple methods exist, explain the advantages and disadvantages of each.
7. Conclude with a final boxed answer (if applicable) for clarity.
"""

# ✅ Initialize Chat Model
chat = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", groq_api_key=API_KEY)

# ✅ Function to Extract Text from Images using OCR.Space API
def extract_text_from_image(image_path):
    try:
        with open(image_path, 'rb') as image_file:
            response = requests.post(
                "https://api.ocr.space/parse/image",
                files={"image": image_file},
                data={"apikey": OCR_API_KEY, "language": "eng"}
            )
        result = response.json()
        return result.get("ParsedResults", [{}])[0].get("ParsedText", "No text detected.")
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        return "Error processing image."

# ✅ Class for Document and Image Processing
class MultiFormatRAG:
    def __init__(self):
        self.loader_map = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.txt': TextLoader,
            '.csv': CSVLoader,
            '.html': UnstructuredHTMLLoader,
            '.md': UnstructuredMarkdownLoader
        }
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    def load_documents(self, directory_path):
        documents = []
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            ext = os.path.splitext(file)[1].lower()
            if ext in self.loader_map:
                try:
                    loader = self.loader_map[ext](file_path)
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    logger.error(f"Error loading {file}: {str(e)}")
        return documents

    def process_documents(self, documents):
        if not documents:
            return None
        texts = self.text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(texts, self.embeddings)
        return vectorstore

    def query(self, qa_chain, question, chat_history):
        response = qa_chain.invoke({"question": question, "chat_history": chat_history})
        return response.get("answer", "No response generated.")

# ✅ Initialize Streamlit Page
st.set_page_config(page_title="AlgebrAI - Math Chatbot", page_icon="🧮", layout="wide")

# ✅ Custom Styling for Chat Messages
st.markdown("""
    <style>
    .chat-container {
        display: flex;
        flex-direction: column;
    }
    .user-message, .ai-message {
        padding: 10px;
        border-radius: 10px;
        max-width: 70%;
        text-align: left;
        clear: both;
        margin: 5px 0;
        display: flex;
        align-items: center;
    }
    .user-message {
        background-color: rgb(241, 234, 26);
        color: black;
        float: right;
    }
    .ai-message {
        background-color: rgb(163, 168, 184);
        color: black;
        float: left;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔢 AlgebrAI - Advanced Math Assistant")
st.write("Ask math questions or upload documents/images for analysis.")

# ✅ Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_system" not in st.session_state:
    st.session_state.rag_system = MultiFormatRAG()
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ✅ Sidebar for File Uploads
with st.sidebar:
    st.title("Upload Files")
    uploaded_files = st.file_uploader("Upload Documents or Images to Extract Text of Maths Problem ", accept_multiple_files=True, type=['pdf', 'docx', 'txt', 'csv', 'html', 'md', 'png', 'jpg', 'jpeg'])

    if uploaded_files and st.button("Process Files"):
        with st.spinner("Processing..."):
            temp_dir = tempfile.mkdtemp()
            text_data = ""  # Store extracted text

            for file in uploaded_files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getvalue())

                # ✅ Extract text if image
                if file.type.startswith('image'):
                    extracted_text = extract_text_from_image(file_path)
                    text_data += extracted_text + "\n"

                # ✅ Extract text if document
                else:
                    documents = st.session_state.rag_system.load_documents(temp_dir)
                    if documents:
                        extracted_text = "\n".join([doc.page_content for doc in documents])
                        text_data += extracted_text + "\n"

                        # ✅ Process into vector embeddings
                        vectorstore = st.session_state.rag_system.process_documents(documents)
                        if vectorstore:
                            st.session_state.vectorstore = vectorstore
                            st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
                                llm=chat,
                                retriever=vectorstore.as_retriever(),
                                memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                            )

            # ✅ Display Extracted Text in Chat
            if text_data.strip():
                formatted_text = "### Extracted Text:\n\n" + text_data.replace("•", "\n-")  # Convert bullet points
                formatted_text = formatted_text.replace("\n", "\n\n")  # Add spacing for better readability
                st.session_state.chat_history.append(AIMessage(content=f"Extracted Text:\n{text_data}"))
                st.success("Text extracted successfully!")

            elif st.session_state.qa_chain:
                st.success("Documents processed successfully!")

# ✅ Display Chat History Properly
chat_container = st.container()
with chat_container:
    for msg in st.session_state.chat_history:
        role = "😀" if isinstance(msg, HumanMessage) else "🤖"
        styled_msg = f"""
            <div class="{'user-message' if role == '😀' else 'ai-message'}">
                <span>{role} : {msg.content}</span>
            </div>
        """
        st.markdown(styled_msg, unsafe_allow_html=True)

# ✅ Chatbot User Input
user_input = st.chat_input("Type your math question...")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # ✅ Display user input immediately
    st.markdown(f"<div class='user-message'><span>😀 : {user_input}</span></div>", unsafe_allow_html=True)

    with st.spinner("Thinking..."):
        response = ""

        # ✅ If RAG-based QA Chain is available, use it
        if st.session_state.qa_chain:
            response = st.session_state.rag_system.query(st.session_state.qa_chain, user_input, st.session_state.chat_history)
        else:
            # ✅ Otherwise, use regular chat model
            full_prompt = [SystemMessage(content=SYSTEM_PROMPT)] + st.session_state.chat_history + [HumanMessage(content=user_input)]
            response = chat.invoke(full_prompt).content

    # ✅ Append AI response and display it immediately
    st.session_state.chat_history.append(AIMessage(content=response))
    st.markdown(f"<div class='ai-message'><span>🤖 : {response}</span></div>", unsafe_allow_html=True)