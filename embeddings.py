from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from config import GOOGLE_API_KEY
from langchain_groq import ChatGroq

chat_model=ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
# Initialize embeddings and chat model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# chat_model = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",
#     google_api_key=GOOGLE_API_KEY
# )
