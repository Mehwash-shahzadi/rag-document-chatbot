import streamlit as st
import logging
import tempfile
import os
import time

from src.document_processor import DocumentProcessor
from src.embeddings import get_embeddings
from src.vector_store import VectorStoreManager
from src.retriever import SemanticRetriever
from src.chatbot import SupportChatbot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🤖",
    layout="centered"
)

# Session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None

def init_chatbot():
    """Initialize chatbot from existing KB"""
    try:
        embeddings = get_embeddings()
        vm = VectorStoreManager()
        vs = vm.load_vector_store(embeddings)
        if vs:
            retriever = SemanticRetriever(vs)
            st.session_state.chatbot = SupportChatbot(vs, retriever)
            return True
    except:
        pass
    return False

def process_docs(files):
    """Process uploaded documents"""
    try:
        with st.status("Processing documents...") as status:
            # Save uploaded files
            temps = []
            for f in files:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{f.name.split('.')[-1]}")
                tmp.write(f.read())
                tmp.close()
                temps.append(tmp.name)
            
            # Process
            processor = DocumentProcessor(temps)
            docs = processor.process()
            
            if not docs:
                status.update(label="Failed to process", state="error")
                return False
            
            # Build KB
            embeddings = get_embeddings()
            vm = VectorStoreManager()
            
            if vm.exists():
                vs = vm.add_documents(docs, embeddings)
            else:
                vs = vm.create_vector_store(docs, embeddings)
            
            # Init chatbot
            retriever = SemanticRetriever(vs)
            st.session_state.chatbot = SupportChatbot(vs, retriever)
            
            # Cleanup
            for tmp in temps:
                try:
                    os.unlink(tmp)
                except:
                    pass
            
            status.update(label="Ready!", state="complete")
            time.sleep(0.5)
        
        return True
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False

# ========================
# SIDEBAR
# ========================
with st.sidebar:
    st.markdown("## 🤖 AI Assistant")
    
    vm = VectorStoreManager()
    
    # Status
    if vm.exists():
        st.success("Active")
    else:
        st.warning("No KB")
    
    st.markdown("---")
    
    # Upload
    st.markdown("#### Upload Files")
    uploaded = st.file_uploader(
        "label",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded:
        if st.button(" Process", use_container_width=True):
            if process_docs(uploaded):
                st.balloons()
                time.sleep(1)
                st.rerun()
    
    st.markdown("---")
    
    # Actions
    if st.button(" New Chat", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.chatbot:
            st.session_state.chatbot.clear_memory()
        st.rerun()
    
    if st.button("Delete KB", use_container_width=True):
        vm.delete_vector_store()
        st.session_state.chatbot = None
        st.session_state.messages = []
        st.rerun()

# ========================
# MAIN
# ========================

# Auto-init if KB exists
if not st.session_state.chatbot and vm.exists():
    init_chatbot()

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Show confidence for assistant messages
        if msg["role"] == "assistant":
            if "conf" in msg:
                c = msg["conf"]
                if c >= 0.7:
                    st.caption(f" {c:.0%}")
                elif c >= 0.5:
                    st.caption(f" {c:.0%}")
                else:
                    st.caption(f" {c:.0%}")
            
            # Sources
            if "src" in msg and msg["src"]:
                with st.expander(" Sources"):
                    st.text(msg["src"])

# Chat input
if st.session_state.chatbot:
    
    if prompt := st.chat_input("Type your question..."):
        
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.chatbot.ask(prompt)
                
                # Show answer
                st.markdown(result['answer'])
                
                # Show confidence
                c = result['confidence']
                if c >= 0.7:
                    st.caption(f" {c:.0%}")
                elif c >= 0.5:
                    st.caption(f" {c:.0%}")
                else:
                    st.caption(f" {c:.0%}")
                
                # Sources
                if result.get('sources'):
                    with st.expander(" Sources"):
                        st.text(result['sources'])
                
                # Save
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result['answer'],
                    "conf": c,
                    "src": result.get('sources', '')
                })
        
        st.rerun()

else:
    # Welcome
    st.markdown("#  AI Knowledge Assistant")
    st.markdown("Chat with your documents using AI")
    
    st.markdown("")
    st.markdown("")
    
    # Features
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("### 🤖")
        st.markdown("**Smart Search**")
        st.markdown("Instant answers")
    
    with c2:
        st.markdown("### 🎯")
        st.markdown("**Confidence**")
        st.markdown("Reliable scores")
    
    with c3:
        st.markdown("### 📚")
        st.markdown("**Sources**")
        st.markdown("See references")
    
    st.markdown("")
    st.markdown("---")
    st.markdown("")
    
    # Guide
    st.markdown("###  Quick Start")
    st.markdown("**1.** Upload PDF or TXT files")
    st.markdown("**2.** Click Process button")
    st.markdown("**3.** Ask your questions!")
    
    st.markdown("")
    st.markdown("")
    
    st.info(" Upload documents in the sidebar to begin")