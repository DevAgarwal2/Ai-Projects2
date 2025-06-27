import streamlit as st
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq
from e2b_code_interpreter import Sandbox
import matplotlib.pyplot as plt
import numpy as np
import base64
import io # Import io for handling bytes

st.set_page_config(page_title="Multi-tool AI Agent", layout="wide")
st.title("🤖 Multi-tool AI Assistant")
st.markdown("Ask me anything! I can execute Python code, perform web searches, and even speak my answers.")

load_dotenv()

@st.cache_resource
def get_groq_chat_client():
    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY")
    )

@st.cache_resource
def get_exa_client():
    return OpenAI(
        base_url="https://api.exa.ai",
        api_key=os.getenv("EXA_API_KEY"),
    )

@st.cache_resource
def get_groq_tts_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

groq_chat_client = get_groq_chat_client()
exa_client = get_exa_client()
groq_tts_client = get_groq_tts_client()

groq_chat_model = "qwen/qwen3-32b"

def execute_python(code: str) -> str:
    try:
        with Sandbox() as sandbox:
            result_text = ""
            generated_image_data_for_ui = []
            
            execution = sandbox.run_code(code)
            
            if execution.error:
                result_text += f"Error during code execution: {str(execution.error.message)}\n"
            elif hasattr(execution.logs, 'logs') and execution.logs.logs:
                log_output = "\n".join([f"[{str(log.level)}] {str(log.line)}" for log in execution.logs.logs])
                result_text += str(execution.text) if execution.text else f"Code executed, output in logs:\n{log_output}\n"
            else:
                result_text += str(execution.text) + "\n"

            if not result_text.strip():
                result_text = "The code executed, but produced no direct output.\n"

            llm_text_summary = ""

            if execution.results:
                image_detected = False
                for i, result_item in enumerate(execution.results):
                    if result_item.png:
                        base64_image_string = f"data:image/png;base64,{result_item.png}"
                        generated_image_data_for_ui.append(base64_image_string)
                        image_detected = True
                    elif result_item.chart:
                        image_detected = True
                
                if image_detected:
                    llm_text_summary = f"Python code executed successfully. A graph has been generated and displayed in the UI. The task of generating the graph is now complete."
                    if result_text.strip():
                        llm_text_summary += f"\nAdditional code output: {result_text.strip()}"
                else:
                    llm_text_summary = "Python code executed, but no direct graph or image output was detected. If you intended to generate a graph, please ensure `plt.savefig()` is used."
                    if result_text.strip():
                        llm_text_summary += f"\nCode output: {result_text.strip()}"
            else:
                llm_text_summary = "Python code executed, but no direct graph or image output was detected. If you intended to generate a graph, please ensure `plt.savefig()` is used."
                if result_text.strip():
                    llm_text_summary += f"\nCode output: {result_text.strip()}"
            
            return json.dumps({
                "text_output": result_text.strip(),
                "image_data": generated_image_data_for_ui,
                "llm_content_summary": {"status": "Tool executed.", "message": llm_text_summary}
            })
    except Exception as e:
        return json.dumps({
            "text_output": f"Failed to execute Python code due to Sandbox initialization/runtime error: {str(e)}",
            "image_data": [],
            "llm_content_summary": {"text_output": f"Error executing Python code: {str(e)}", "image_info": "No images generated due to error."}
        })

def realtime_search(query: str) -> str:
    try:
        search_response = exa_client.chat.completions.create(
            model="exa",
            messages=[
                {"role": "user", "content": query}
            ],
            extra_body={"text": True}
        )
        content = search_response.choices[0].message.content
        citations = search_response.choices[0].message.citations
        
        citations_str = ""
        if citations:
            citations_str = "\n\nSources:\n"
            for i, citation in enumerate(citations):
                title = citation.get('title', 'No Title')
                url = citation.get('url', 'No URL')
                citations_str += f"{i+1}. [{title}]({url})\n"
        
        llm_summary = f"Web search for '{query}' completed. Found relevant information."
        if citations:
            llm_summary += f" ({len(citations)} source(s) found)."

        return json.dumps({
            "full_content": f"Search Result: {content}{citations_str}",
            "llm_content_summary": llm_summary
        })
    except Exception as e:
        llm_summary = f"Error performing real-time search for '{query}': {str(e)}"
        return json.dumps({
            "full_content": llm_summary,
            "llm_content_summary": llm_summary
        })

def text_to_speech(text: str, file_name: str = "speech_output.wav") -> str:
    try:
        tts_response = groq_tts_client.audio.speech.create(
            model="playai-tts",
            voice="Fritz-PlayAI",
            input=text,
            response_format="wav"
        )
        
        # Get audio content directly as bytes
        audio_bytes = tts_response.read()
        
        # Encode bytes to base64 string for JSON serialization
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        llm_summary = f"Speech successfully generated. Audio is embedded and playable in the UI."
        return json.dumps({
            "full_message": "Speech successfully generated and is available for playback.",
            "audio_base64": audio_base64, # Store base64 string for direct playback in Streamlit
            "llm_content_summary": llm_summary
        })
    except Exception as e:
        llm_summary = f"Error generating speech for text '{text}': {str(e)}"
        return json.dumps({
            "full_message": llm_summary,
            "audio_base64": None,
            "llm_content_summary": llm_summary
        })

tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Executes Python code in a secure sandboxed environment. Use this tool for calculations, data processing, logic testing, or any task that requires code execution. It can also save and return paths to generated image files (like plots).",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute in a single cell. If generating a graph, ensure `plt.savefig('filename.png')` is used to save the image (e.g., to `/home/user/filename.png`), and `plt.show()` is NOT used. E2B will automatically return saved plots as PNGs."
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "realtime_search",
            "description": "Performs a real-time web search to find up-to-date information, news, definitions, or facts not available in your training data. Always use this for queries about current events, recent developments, or specific factual lookups. Returns a summary and sources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to use for the web search."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "text_to_speech",
            "description": "Converts specified text into an audio (.wav) file. Use this when the user explicitly asks you to 'say' something, 'generate audio', or provides text to be spoken. You can also specify a file name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text content that needs to be converted into speech."
                    },
                    "file_name": {
                        "type": "string",
                        "description": "Optional: The desired name for the output audio file (e.g., 'hello_message.wav'). Defaults to 'speech_output.wav'."
                    }
                },
                "required": ["text"]
            }
        }
    }
]

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a versatile AI assistant capable of executing Python code, performing real-time web searches, and converting text to speech. Analyze user requests to determine the most appropriate tool or combination of tools to achieve the desired outcome. If a request involves getting data and then generating a graph, use the search tool first, then the Python interpreter to generate the graph. Once the graph is successfully generated and displayed, provide a concise summary of the graph and the data, and conclude the conversation."}
    ]

if "show_quick_actions_section" not in st.session_state:
    st.session_state.show_quick_actions_section = True

def run_conversation(user_input_text):
    st.session_state.messages.append({"role": "user", "content": user_input_text})
    with st.chat_message("user"):
        st.markdown(user_input_text)

    with st.chat_message("assistant"):
        temp_messages = list(st.session_state.messages) 

        with st.spinner("Assistant is thinking and interacting..."):
            while True:
                response = groq_chat_client.chat.completions.create(
                    model=groq_chat_model,
                    messages=temp_messages,
                    tools=tools,
                    tool_choice="auto"
                )

                response_message = response.choices[0].message
                
                if response_message.tool_calls:
                    st.info("Assistant decided to call tools:")
                    tool_call_display = json.dumps([tc.model_dump() for tc in response_message.tool_calls], indent=2)
                    st.code(tool_call_display, language="json")
                    
                    for tool_call in response_message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        tool_output_json_str = ""

                        st.info(f"Executing tool: **`{tool_name}`** with arguments: `{tool_args}`")

                        if tool_name == "execute_python":
                            tool_output_json_str = execute_python(tool_args.get('code', ''))
                            tool_output_data = json.loads(tool_output_json_str)
                            
                            st.code(tool_output_data["text_output"], language="python") 
                            for i, img_base64_data in enumerate(tool_output_data.get("image_data", [])):
                                st.image(img_base64_data, caption="Generated Plot")
                                st.success(f"Plot {i+1} generated and displayed.")
                            
                            content_for_llm = json.dumps(tool_output_data.get("llm_content_summary", {"status": "Tool executed."}))


                        elif tool_name == "realtime_search":
                            tool_output_json_str = realtime_search(tool_args.get('query', ''))
                            tool_output_data = json.loads(tool_output_json_str)

                            st.markdown(tool_output_data["full_content"])
                            
                            content_for_llm = tool_output_data.get("llm_content_summary", "Search completed.")


                        elif tool_name == "text_to_speech":
                            tts_text = tool_args.get('text', '')
                            tts_file_name = tool_args.get('file_name', 'speech_output.wav')
                            tool_output_json_str = text_to_speech(tts_text, tts_file_name)
                            tool_output_data = json.loads(tool_output_json_str)

                            st.markdown(tool_output_data["full_message"])
                            # Decode base64 string to bytes for Streamlit audio widget
                            if tool_output_data.get("audio_base64"):
                                audio_bytes_decoded = base64.b64decode(tool_output_data["audio_base64"])
                                st.audio(audio_bytes_decoded, format="audio/wav")
                                st.success(f"Audio generated and played.")
                            else:
                                st.error("Failed to play audio.")
                            
                            content_for_llm = tool_output_data.get("llm_content_summary", "Text to speech completed.")

                        else:
                            tool_output_json_str = json.dumps({"text_output": f"Unknown tool requested: {tool_name}", "llm_content_summary": f"Unknown tool: {tool_name}"})
                            tool_output_data = json.loads(tool_output_json_str)
                            st.warning(tool_output_data["text_output"])
                            content_for_llm = tool_output_data["llm_content_summary"]
                        
                        temp_messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_name,
                            "content": content_for_llm,
                        })
                else:
                    final_assistant_message = response_message.content
                    st.markdown(final_assistant_message)
                    temp_messages.append({"role": "assistant", "content": final_assistant_message})
                    break

            st.session_state.messages = temp_messages

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            if message["role"] != "tool":
                st.markdown(message["content"])
            else:
                st.info(f"Tool `{message['name']}` executed.")
                try:
                    tool_content_data = json.loads(message["content"])
                    
                    if message["name"] == "execute_python":
                        st.code(tool_content_data.get("text_output", ""), language="python")
                        for i, img_base64_data in enumerate(tool_content_data.get("image_data", [])):
                            st.image(img_base64_data, caption="Generated Plot")
                    elif message["name"] == "realtime_search":
                        st.markdown(tool_content_data.get("full_content", "Search executed."))
                    elif message["name"] == "text_to_speech":
                        st.markdown(tool_content_data.get("full_message", "Text to speech executed."))
                        # Check for audio_base64 instead of audio_path
                        if tool_content_data.get("audio_base64"):
                            audio_bytes_decoded = base64.b64decode(tool_content_data["audio_base64"])
                            st.audio(audio_bytes_decoded, format="audio/wav")
                    else:
                        st.markdown(message["content"])

                except json.JSONDecodeError:
                    st.markdown(message["content"])
                
st.markdown("---")

if st.session_state.show_quick_actions_section:
    st.subheader("Quick Actions:")
    col1, col2, col3 = st.columns(3)

    PROMPTS = {
        "Search for News": "Search for the latest news on [topic] (e.g., 'India Rath Yatra news')",
        "Generate Graph": "Generate a graph of india gpd growth rate for last 5 years",
        "Text to Speech": "Say hello from your AI assistant "
    }

    with col1:
        if st.button("🔍 Search for News", use_container_width=True):
            st.session_state.show_quick_actions_section = False
            run_conversation(PROMPTS["Search for News"])
    with col2:
        if st.button("📊 Generate Graph", use_container_width=True):
            st.session_state.show_quick_actions_section = False
            run_conversation(PROMPTS["Generate Graph"])
    with col3:
        if st.button("🔊 Text to Speech", use_container_width=True):
            st.session_state.show_quick_actions_section = False
            run_conversation(PROMPTS["Text to Speech"])
else:
    if st.button("Start New Conversation / Show Quick Actions", use_container_width=True):
        st.session_state.messages = [
            {"role": "system", "content": "You are a versatile AI assistant capable of executing Python code, performing real-time web searches, and converting text to speech. Analyze user requests to determine the most appropriate tool or combination of tools to achieve the desired outcome. If a request involves getting data and then generating a graph, use the search tool first, then the Python interpreter to generate the graph. Once the graph is successfully generated and displayed, provide a concise summary of the graph and the data, and conclude the conversation."}
        ]
        st.session_state.show_quick_actions_section = True
        st.rerun()

user_query = st.chat_input("Enter your query here...")
if user_query:
    st.session_state.show_quick_actions_section = False
    run_conversation(user_query)
