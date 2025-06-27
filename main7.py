import streamlit as st
import os
from dotenv import load_dotenv
import PyPDF2
import docx
import time
import random
from sarvamai import SarvamAI

load_dotenv()

@st.cache_resource
def initialize_clients():
    """Initialize SarvamAI client"""
    return SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        return "\n".join(page.extract_text() for page in pdf_reader.pages)
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def extract_text_from_docx(docx_file):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(docx_file)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return None

def extract_text_from_txt(txt_file):
    """Extract text from TXT file"""
    try:
        return txt_file.read().decode('utf-8')
    except Exception as e:
        st.error(f"Error reading TXT: {str(e)}")
        return None

def translate_text(client, text, target_language):
    """Translate text using SarvamAI with improved error handling"""
    try:
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        translated_chunks = []
        
        for chunk in chunks:
            try:
                response = client.text.translate(
                    input=chunk,
                    source_language_code="en-IN",
                    target_language_code=target_language
                )
                translated_chunks.append(response.translated_text)
            except Exception as chunk_error:
                st.warning(f"Skipped a chunk due to translation error")
                translated_chunks.append(chunk)
                continue
                
        return " ".join(translated_chunks)
        
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text

def generate_answer(sarvam_client, context, question):
    """Generate answer using SarvamAI"""
    try:
        # Trim context if too large
        max_context_length = 4000
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."

        prompt = f"""Based on the following medical document context, please answer the question accurately and concisely.

Context: {context}

Question: {question}

Answer:"""
        
        for attempt in range(3):
            try:
                response = sarvam_client.chat.completions(messages=[
                    {"role": "user", "content": prompt}
                ])
                return response.choices[0].message.content
            except Exception as e:
                if attempt < 2:  # Don't sleep on last attempt
                    time.sleep((2 ** attempt) + random.uniform(0, 1))
                continue
                
        return "Sorry, I couldn't generate an answer. Please try again."
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return "Sorry, I couldn't process your question. Please try again."

def main():
    st.set_page_config(
        page_title="Medical Document Translator & Q&A",
        page_icon="🏥",
        layout="centered"
    )
    
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
            margin-top: 10px;
            background-color: #4CAF50;
            color: white;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            border: 1px solid #e0e0e0;
        }
        .chat-question {
        }
        .chat-answer {
            
            margin-left: 1rem;
        }
        .st-emotion-cache-1v0mbdj.e115fcil1 {
            border: 1px solid #e0e0e0;
            border-radius: 0.5rem;
            padding: 1rem;
        }
        .success-message {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🏥 Medical Document Translator")
    st.markdown("_Translate medical documents and get instant answers to your questions_")
    
    st.markdown(" 🌐 Select Language")
    language_options = {
        'Hindi': 'hi-IN',
        'English': 'en-IN',
        'Bengali': 'bn-IN',
        'Tamil': 'ta-IN',
        'Telugu': 'te-IN',
        'Marathi': 'mr-IN',
        'Gujarati': 'gu-IN',
        'Kannada': 'kn-IN',
        'Malayalam': 'ml-IN',
        'Punjabi': 'pa-IN',
        'Urdu': 'ur-IN'
    }
    
    col1, col2 = st.columns([2,1])
    with col1:
        selected_language = st.selectbox(
            "Target Language",
            options=list(language_options.keys()),
            index=0
        )
    
    target_language_code = language_options[selected_language]
    
    st.markdown("### 📄 Upload Document")
    uploaded_file = st.file_uploader(
        "",  
        type=['pdf', 'docx', 'txt'],
        help="Upload your medical document (PDF, DOCX, or TXT format)"
    )
    
    if uploaded_file is not None:
        # Process uploaded file
        file_processors = {
            "application/pdf": extract_text_from_pdf,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": extract_text_from_docx,
            "text/plain": extract_text_from_txt
        }
        
        processor = file_processors.get(uploaded_file.type)
        if processor:
            extracted_text = processor(uploaded_file)
            if extracted_text:
                with st.expander("📝 Original Text", expanded=False):
                    st.text_area("", extracted_text, height=150)
                
                if st.button("🔄 Translate", use_container_width=True):
                    with st.spinner(f"Translating to {selected_language}..."):
                        translated_text = translate_text(initialize_clients(), extracted_text, target_language_code)
                        
                        if translated_text:
                            st.session_state['translated_text'] = translated_text
                            st.session_state['original_text'] = extracted_text
                            st.session_state['document_chunks'] = [f"{extracted_text}\n\n{translated_text}"]
                            
                            with st.expander(f"📄 {selected_language} Translation", expanded=True):
                                st.text_area("", translated_text, height=200)
                            
                            st.markdown('<div class="success-message">✅ Document translated successfully!</div>', unsafe_allow_html=True)
    
    # Q&A Section
    if 'document_chunks' in st.session_state:
        st.markdown("---")
        st.markdown("❓ Ask Questions")
        
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        
       
        user_question = st.text_input("", placeholder="Ask about the document...", key="user_question")
        
        if st.button("🔍 Get Answer", use_container_width=True) and user_question:
            with st.spinner("Finding answer..."):
                answer = generate_answer(initialize_clients(), st.session_state['document_chunks'][0], user_question)
                st.session_state.chat_history.append((user_question, answer))
        
        
        if st.session_state.chat_history:
            st.markdown(" Previous Questions & Answers")
            for question, answer in reversed(st.session_state.chat_history):
                st.markdown(f'<div class="chat-message chat-question">❓ {question}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-message chat-answer">💡 {answer}</div>', unsafe_allow_html=True)
    
    else:
        st.info("👆 Upload and translate a document to start asking questions")

if __name__ == "__main__":
    main()