import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, dotenv_values

# Laden der Umgebungsvariablen
load_dotenv(".env")
env_vars = dotenv_values()
api_key = env_vars.get("OPEN_AI_API_KEY")

# Initialisierung des LLM und der Embeddings
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

# Initialisierung des Vektorspeichers
vector_store = InMemoryVectorStore(embeddings)

# Chatverlauf s
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Funktion zum Laden und Verarbeiten von PDF-Dateien
def process_pdfs(uploaded_files):
    all_splits = []
    for uploaded_file in uploaded_files:
        # Speichern der pdfs
        file_path = f"uploaded_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Laden und Verarbeiten der PDF-Datei
        loader = PyPDFLoader(file_path)
        pages = []
        for page in loader.load():
            pages.append(page)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits.extend(text_splitter.split_documents(pages))
    
    # die chunks in den Vektorstore speichern
    vector_store.add_documents(documents=all_splits)

# Funktion zur Abfrage des Vektorspeichers
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=3)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized

# Generierung der Antwort
def generate_answer(user_query: str):
    """Generates an answer to the user query using the model."""
    # Suche relevante Chunks im Vektorspeicher
    retrieved_docs = retrieve(user_query)
    
    # Systemprompt
    system_message_content = (
        """Du bist ein Assistent, der Fragen über die Inhalte von PDFs beantwortet.
        Wenn du etwas nicht beantworten kannst, gib bitte an, dass du es nicht weißt. Beschränke deine Antworten auf höchstens drei Sätze und bleibe möglichst prägnant.""" 
        "\n\n"
        f"Relevante Abschnitte: \n{retrieved_docs}"
    )
    
    response = llm.invoke([{"role": "system", "content": system_message_content}, {"role": "user", "content": user_query}])
    
    # Zugriff auf den Inhalt der Antwort
    return response.content

# Streamlit-Interface
st.title("Langchain PDF RAG")

# Mehrere PDF-Dateien hochladen
uploaded_files = st.file_uploader("Lade eine oder mehrere PDFs hoch", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    process_pdfs(uploaded_files)
    st.write("Daten verarbeitet und in der Vektordatenbank gespeichert.")
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_query = st.chat_input("Stelle eine Frage über den PDF-Inhalt:")

    if user_query:
        answer = generate_answer(user_query)

        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.messages.append({"role": "assistant", "content": answer})

        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(answer)
else:
    st.write("Bitte lade eine oder mehrere PDFs hoch, um mit der Verarbeitung zu beginnen.")
