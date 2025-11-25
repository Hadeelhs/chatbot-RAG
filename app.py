import streamlit as st
import psycopg
import os
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


# Configuration de la page
st.set_page_config(
    page_title="Chatbot",
    page_icon="💬",
    layout="centered"
)

# Paramètres de connexion
DB_NAME = "rag_chatbot"
DB_USER = "postgres"
DB_PASSWORD = "1234"
DB_HOST = "localhost"
DB_PORT = "5433"
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
db_connection_str = f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} host={DB_HOST} port={DB_PORT}"

# Initialisation des modèles (mis en cache)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def initialize_groq_client():
    return Groq(api_key=GROQ_API_KEY)

embedding_model = load_embedding_model()
groq_client = initialize_groq_client()


def calculate_embeddings(corpus: str) -> list[float]:
    """Générer des embeddings pour un texte."""
    embedding = embedding_model.encode(corpus, convert_to_numpy=True)
    return embedding.tolist()


def similar_corpus(input_corpus: str, top_k: int = 3) -> list[tuple[int, str, float]]:
    """Trouver les entrées similaires dans la base de données."""
    try:
        input_embedding = calculate_embeddings(input_corpus)
        
        with psycopg.connect(db_connection_str) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, corpus, embedding <=> %s::vector AS distance
                    FROM embeddings
                    ORDER BY distance
                    LIMIT %s
                    """,
                    (input_embedding, top_k)
                )
                results = cur.fetchall()
                return results
    except Exception as e:
        print(f"Erreur de connexion DB : {e}")
        return []


def query_with_context(question: str) -> tuple[str, list]:
    """Interroger le chatbot avec RAG. Retourne (réponse, sources)."""
    # Trouver les entrées similaires
    similar_entries = similar_corpus(question, top_k=3)
    
    if not similar_entries:
        return "Désolé, je n'ai pas pu trouver d'informations pertinentes.", []
    
    # Construire le contexte
    context = "\n".join([entry[1] for entry in similar_entries])
    
    # Créer le prompt
    prompt = f"""En te basant sur le contexte de conversation suivant, réponds à la question en français de manière claire et précise.

Contexte :
{context}

Question : {question}

Réponse :"""
    
    # Générer la réponse
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=1024,
        )
        response = chat_completion.choices[0].message.content
        return response, similar_entries
    except Exception as e:
        print(f"Erreur détaillée : {e}")
        error_msg = f"Erreur lors de la génération de la réponse. Vérifiez votre clé API Groq."
        return error_msg, []


# CSS personnalisé pour un style épuré
st.markdown("""
    <style>
    .main {
        max-width: 800px;
        margin: 0 auto;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Titre simple
st.title("💬 Vous avez une question?")

# Initialiser l'historique des messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher l'historique des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Afficher les références pour l'assistant
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("📚 Voir les sources", expanded=False):
                for i, (id, corpus, distance) in enumerate(message["sources"], 1):
                    similarity = (1 - distance) * 100
                    st.markdown(f"**Source {i}** • Pertinence: {similarity:.1f}%")
                    st.caption(corpus[:300] + "..." if len(corpus) > 300 else corpus)
                    if i < len(message["sources"]):
                        st.divider()

# Input utilisateur
if prompt := st.chat_input("Écrivez votre message..."):
    # Ajouter et afficher le message de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Générer et afficher la réponse
    with st.chat_message("assistant"):
        with st.spinner(""):
            response, sources = query_with_context(prompt)
        st.markdown(response)
        
        # Afficher les sources/références
        if sources:
            with st.expander("📚 Voir les sources", expanded=False):
                for i, (id, corpus, distance) in enumerate(sources, 1):
                    similarity = (1 - distance) * 100
                    st.markdown(f"**Source {i}** • Pertinence: {similarity:.1f}%")
                    st.caption(corpus[:300] + "..." if len(corpus) > 300 else corpus)
                    if i < len(sources):
                        st.divider()
    
    # Ajouter la réponse à l'historique
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "sources": sources
    })