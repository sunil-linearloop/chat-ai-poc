import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler
import tiktoken
import os
import psutil
import time

st.set_page_config(page_title="LangChain OpenAI", page_icon=":robot_face:")

os.environ["OPENAI_API_KEY"] = st.secrets['token']

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

chat = ChatOpenAI(
    model_name="gpt-4o-mini-2024-07-18",
    temperature=0.7,
    max_tokens=4096,
    streaming=True
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a highly capable AI assistant powered by GPT-4. You are designed to provide accurate, comprehensive, and precise information on a wide range of topics. Your responses should be thoughtful, well-structured, and tailored to the user's needs."),
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
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

# Chat interface
st.title("LangChain OpenAI GPT-4 Chat Application")

for message in st.session_state.memory.chat_memory.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# React to user input
if user_input := st.chat_input("What is your message?"):
    # Record initial memory usage
    initial_memory = get_memory_usage()

    # Display user message in chat message container
    st.chat_message("human").markdown(user_input)

    # Calculate input tokens
    input_text = "\n".join([msg.content for msg in st.session_state.memory.chat_memory.messages]) + "\n" + user_input
    input_tokens = count_tokens(input_text)

    with st.chat_message("ai"):
        stream_handler = StreamHandler(st.empty())
        start_time = time.time()
        full_response = conversation.run(input=user_input, callbacks=[stream_handler])
        end_time = time.time()

        # Calculate output tokens
        output_tokens = count_tokens(full_response)

        # Get final memory usage
        final_memory = get_memory_usage()

        # Display estimated token usage and memory usage
        st.write(f"Estimated token usage - Input: {input_tokens}, Output: {output_tokens}, Total: {input_tokens + output_tokens}")
        st.write(f"Memory usage - Initial: {initial_memory:.2f} MB, Final: {final_memory:.2f} MB, Difference: {final_memory - initial_memory:.2f} MB")
        st.write(f"Processing time: {end_time - start_time:.2f} seconds")

# Print memory contents (for debugging)
# print(st.session_state.memory.chat_memory.messages)
