import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler
import tiktoken
import os
import psutil
import time

st.set_page_config(page_title="LangChain Groq", page_icon=":robot_face:")

os.environ["GROQ_API_KEY"] = st.secrets['token']

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

st.title("LangChain Groq Chat Application")

# Add model selection dropdown
model_options = ["llama-3.1-70b-versatile", "llama-3.1-8b-instant"]

selected_model = st.selectbox("Select a model:", model_options)

# Update chat initialization with selected model
chat = ChatGroq(
    model_name=selected_model,
    temperature=0.7,
    max_tokens=4096,
    streaming=True
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a strictly obedient and detail-oriented AI assistant. Whatever the user asks, you must execute exactly as instructed, with no exceptions, shortcuts, or omissions. You are bound to follow every command without hesitation or laziness. Your responses must be flawless, providing accurate, comprehensive, and precise information as if you were being paid $5,000 for absolute perfection. No task or detail is too small or insignificant to skip."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

conversation = ConversationChain(
    llm=chat,
    memory=st.session_state.memory,
    prompt=prompt,
    # verbose=True
)

def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Use a similar encoding
    return len(encoding.encode(text))

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

for message in st.session_state.memory.chat_memory.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

if user_input := st.chat_input("What is your message?"):
    initial_memory = get_memory_usage()

    st.chat_message("human").markdown(user_input)

    input_text = "\n".join([msg.content for msg in st.session_state.memory.chat_memory.messages]) + "\n" + user_input
    input_tokens = count_tokens(input_text)

    with st.chat_message("ai"):
        stream_handler = StreamHandler(st.empty())
        start_time = time.time()
        full_response = conversation.run(input=user_input, callbacks=[stream_handler])
        end_time = time.time()

        output_tokens = count_tokens(full_response)

        final_memory = get_memory_usage()

        st.write(f"Estimated token usage - Input: {input_tokens}, Output: {output_tokens}, Total: {input_tokens + output_tokens}")
        # st.write(f"Memory usage - Initial: {initial_memory:.2f} MB, Final: {final_memory:.2f} MB, Difference: {final_memory - initial_memory:.2f} MB")
        st.write(f"Processing time: {end_time - start_time:.2f} seconds")

    # print(st.session_state.memory.chat_memory.messages)
