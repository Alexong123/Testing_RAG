import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css
import pickle
from langchain_community.vectorstores import FAISS
import ollama
from openai import OpenAI

# Load the PDF file in vector store
def load_processed_docs():
    with open('processed_docs.pkl', 'rb') as f:
        return pickle.load(f)

# Load the database from MySQL Workbench
def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_url = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_url)

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)

    template = """
        You are querying a database of Christian documents. Based on the SQL query and its results, provide a concise answer.

        SQL Query: {query}
        Query Result: {response}
        User Question: {question}

        Rules:
        1. If the result is a list of titles, respond with just the titles separated by commas, without any prefix.
        2. Do not add any explanations or prefixes.
        3. If no results are found, say "No matching documents found."
        4. Only return titles that are exact matches from the query result.
        5. If the query doesn't match any documents or the question can't be answered with the available data, respond with "No relevant information found in the database."

        Your response:
        """
    
    prompt = ChatPromptTemplate.from_template(template)

    llm = Ollama(model="llama3")

    def format_sql_response(response):
        if isinstance(response, list):
            if len(response) == 0:
                return "No matching documents found"
            if 'record_count' in response[0]:
                return str(response[0]['record_count'])
            return ", ".join(item['Title'] for item in response if 'Title' in item)
        return str(response)

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            response=lambda vars: format_sql_response(db.run(vars["query"])),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

def get_sql_chain(db):
    template = """
        You are querying a database of Christian documents. Write a SQL query to answer the user's question based on the schema below:

        <SCHEMA>{schema}</SCHEMA>

        Important: 
        1. Use LIKE with wildcards (%) for text searches in Title, Summary, and Keyword fields.
        2. Use SELECT DISTINCT Title FROM christian.christiandatabase as the base query.
        3. Combine multiple search terms with OR conditions.
        4. Break down complex questions into key concepts and search for each.
        5. Write only the SQL query, nothing else.

        Examples:
        1. Question: Who is Jesus?
           SQL Query: SELECT DISTINCT Title FROM christian.christiandatabase WHERE Summary LIKE '%Jesus%' OR Keyword LIKE '%Jesus%' OR Title LIKE '%Jesus%';

        2. Question: Why do we need Christian?
           SQL Query: SELECT DISTINCT Title FROM christian.christiandatabase WHERE Summary LIKE '%need%' OR Summary LIKE '%Christian%' OR Keyword LIKE '%Christian%' OR Title LIKE '%Christian%';
        
        3. Question: Can you explain to me about catechism?
           SQL Query: SELECT DISTINCT Title FROM christian.christiandatabase WHERE Summary LIKE '%Catechism%' OR Keyword LIKE '%Catechism%' OR Title LIKE '%Catechism%';

        4. Question: What does the Bible say about salvation and eternal life?
           SQL Query: SELECT DISTINCT Title FROM christian.christiandatabase WHERE Summary LIKE '%salvation%' OR Summary LIKE '%eternal life%' OR Keyword LIKE '%salvation%' OR Keyword LIKE '%eternal life%' OR Title LIKE '%salvation%' OR Title LIKE '%eternal life%';

        Question: {question}
        SQL Query:
        """
    
    prompt = ChatPromptTemplate.from_template(template)

    llm = Ollama(model="llama3")
    
    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_conversation_chain(vectorstore, selected_model):
    llm = Ollama(model=selected_model)
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True,
        output_key='answer'
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
        output_key='answer'
    )
    return conversation_chain

def handle_userinput(user_question, isSuccess):
    if isSuccess:
        response = st.session_state.conversation({
            "question": user_question,
        })
        
        bot_response_content = response['answer']
        
        source_documents = response.get('source_documents', [])
        sources = list(set([doc.metadata.get('source', 'Unknown source') for doc in source_documents]))
        
        if sources:
            source_text = ", ".join(sources)
            response_with_source = f"{bot_response_content}\n\n_Citation: {source_text}_"
        else:
            response_with_source = f"{bot_response_content}\n\n_No specific source available_"
        
        st.markdown(response_with_source)
        st.session_state.chat_history.append(AIMessage(content=response_with_source))
        
        if source_documents:
            st.subheader("Sources")
            for i, doc in enumerate(source_documents):
                source = doc.metadata.get('source', 'Unknown source')
                with st.expander(f"Source {i + 1}: {source}"):
                    st.write(doc.page_content)
    else:
        st.markdown("No document found to reply. Please ask another question related.")
        st.session_state.chat_history.append(AIMessage(content="No document found to reply. Please ask another question related."))

def extract_model_names(models_info: list, predefined_models: list) -> tuple:
    return tuple(model["name"] for model in models_info["models"] if model["name"] in predefined_models)

allowed_files = ["A Baptist Catechism", 
                 "AnglicanCatechism",
                 "AthanasionCreed",
                 "Basics of Christianity",
                 "Chrisitian_merged",
                 "Christian Dogmatics_text",
                 "essays-in-evangelical-social-ethics",
                 "Evangelical Ethics_ Issues Facing the Church Today, 4th ed."
                 ]

def process_file_names(input_string, allowed_files):
    processed_files = []
    
    # Split the input string by comma and strip whitespace
    file_names = [name.strip() for name in input_string.split(',')]

    print(file_names)
    
    for file_name in file_names:
        if file_name in allowed_files:
            final_name = file_name + ".pdf"
            processed_files.append(f"'{final_name}'")
    
    print(processed_files)
    
    return processed_files

def main():
    load_dotenv()

    st.set_page_config(
        page_title="Chat with Christian Documents", 
        page_icon=":computer:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_docs" not in st.session_state:
        st.session_state.processed_docs = load_processed_docs()
    
    st.subheader("Chat with Christian Documents :computer:",divider="red",anchor=False)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage("Hello! I'm a SQL assistant. Ask me anything about your database."),
        ]

    client = OpenAI(
        base_url="http://127.0.0.1:11434/",
        api_key="ollama",
    )

    models_info = ollama.list()
    predefined_models = ["llama3:latest", "qwen2:latest"]
    available_models = extract_model_names(models_info, predefined_models)

    with st.sidebar:
        st.subheader("Model Selection")
        
        if available_models:
            selected_model = st.sidebar.selectbox(
                "Pick a model available locally on your system ↓", available_models
            )
        else:
            st.sidebar.warning("You have not pulled any model from Ollama yet!", icon="⚠️")
            selected_model = None
    
        st.subheader("Database Connection")
        st.write("Press the button to connect to the database.")
        st.text_input("Host", value="localhost", key="Host")
        st.text_input("Port", value="3306", key="Port")
        st.text_input("User", value="root", key="User")
        st.text_input("Password", type="password", value="admin", key="Password")
        st.text_input("Database", value="christian", key="Database")

        if st.button("Connect Database"):
            with st.spinner("Loading all documents..."):
                st.session_state.db = init_database(
                    st.session_state["User"],
                    st.session_state["Password"],
                    st.session_state["Host"],
                    st.session_state["Port"],
                    st.session_state["Database"]
                )
                st.success("Connected to database")

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)

    user_question = st.chat_input("Type a message...")

    if user_question:
        if st.session_state.db is None:
            st.error("Please load a document or all documents before asking questions!!!")
        else:
            st.session_state.chat_history.append(HumanMessage(content=user_question))

        with st.chat_message("Human"):
            st.markdown(user_question)

        with st.chat_message("AI"):
            doc_title = get_response(user_question, st.session_state.db, st.session_state.chat_history)
            print("Doc founded:", doc_title)

            processed_files = process_file_names(doc_title, allowed_files)

            processed_file_names = [file.strip("'") for file in processed_files]

            print(process_file_names)

            combined_vectorstore = None
            for doc_title in processed_file_names:
                if doc_title in st.session_state.processed_docs:
                    vectorstore = st.session_state.processed_docs[doc_title]['vectorstore']
                    if combined_vectorstore is None:
                        combined_vectorstore = vectorstore
                    else:
                        combined_vectorstore.merge_from(vectorstore)

            if combined_vectorstore is not None:
                print(combined_vectorstore)
                st.session_state.conversation = get_conversation_chain(combined_vectorstore, selected_model)
                handle_userinput(user_question,True)
            else:
                handle_userinput(user_question,False)

if __name__ == "__main__":
    main()
