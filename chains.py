from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from embeddings import chat_model, embeddings
from langchain.chains import create_history_aware_retriever

# Define your prompt templates
prompt_template = ChatPromptTemplate.from_template(
    """
please act like a conversational ai and responce should be concise.if user say hi respond hello how can i help you like this
please give the answer 2 3 line every time
    <context>
    {context}
    </context>
    
    Question: {input}"""
)

system_prompt = (
    """
please act like a conversational ai and responce should be concise.if user say hi respond hello how can i help you like this
please give me the answer 2 3 line evry time
"""
    
    "{context}"
)

# Load your vectorstore
vector_store = FAISS.load_local(r"faissindexupdate10", embeddings, allow_dangerous_deserialization=True)

# Create document chain
document_chain = create_stuff_documents_chain(chat_model, prompt_template)

# Create retriever and retrieval chain
retriever = vector_store.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Create history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    chat_model, retriever, 
    ChatPromptTemplate.from_messages(
        [
            ("system", "Given a chat history and the latest user question..."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
)

# Create final question-answering chain
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(chat_model, qa_prompt)

# Create RAG chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)