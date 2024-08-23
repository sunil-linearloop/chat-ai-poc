import streamlit as st
import groq
import os
import time
import tiktoken
import psutil

st.set_page_config(page_title="Chat Groq", page_icon=":robot_face:")

# Set up Groq API key
os.environ["GROQ_API_KEY"] = st.secrets['token']

st.title("Traditional Groq Chat Application")

model_options = ["llama-3.1-70b-versatile", "llama-3.1-8b-instant"]

selected_model = st.selectbox("Select a model:", model_options)

# Initialize Groq client
client = groq.Groq()

# Function to count tokens
def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Use a similar encoding
    return len(encoding.encode(text))

# Function to get current memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat interface

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if user_input := st.chat_input("What is your message?"):
    # Record initial memory usage
    initial_memory = get_memory_usage()

    # Display user message in chat message container
    st.chat_message("user").markdown(user_input)

    # Append user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Prepare the messages for the API call
    api_messages = [
        {"role": "system", "content": "You are a strictly obedient and detail-oriented AI assistant. Whatever the user asks, you must execute exactly as instructed, with no exceptions, shortcuts, or omissions. You are bound to follow every command without hesitation or laziness. Your responses must be flawless, providing accurate, comprehensive, and precise information as if you were being paid $5,000 for absolute perfection. No task or detail is too small or insignificant to skip."},
    ] + st.session_state.messages

    # Calculate input tokens
    input_text = "\n".join([msg["content"] for msg in api_messages])
    input_tokens = count_tokens(input_text)

    # Make API call to Groq
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        start_time = time.time()
        
        for chunk in client.chat.completions.create(
            model=selected_model,
            messages=api_messages,
            max_tokens=4096,
            temperature=0.7,
            stream=True
        ):
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        end_time = time.time()

        # Append assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Calculate output tokens
        output_tokens = count_tokens(full_response)

        # Get final memory usage
        final_memory = get_memory_usage()

        # Display estimated token usage and memory usage
        st.write(f"Estimated token usage - Input: {input_tokens}, Output: {output_tokens}, Total: {input_tokens + output_tokens}")
        # st.write(f"Memory usage - Initial: {initial_memory:.2f} MB, Final: {final_memory:.2f} MB, Difference: {final_memory - initial_memory:.2f} MB")
        st.write(f"Processing time: {end_time - start_time:.2f} seconds")
