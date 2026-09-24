import streamlit as st
import yaml
from pathlib import Path

# 1. Page Configuration MUST be the first command
st.set_page_config(
    page_title="PhytoVision AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Global State Initialization
def init_session_state():
    if 'config' not in st.session_state:
        config_path = Path(__file__).parent / "assets" / "config.yaml"
        with open(config_path, encoding='utf-8') as f:
            st.session_state.config = yaml.safe_load(f)
            
    if 'language' not in st.session_state:
        st.session_state.language = st.session_state.config['languages'][0]['code']

# 3. UI Asset Loading
@st.cache_data
def load_css():
    css_path = Path(__file__).parent / "utils" / "style.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 4. Main Application Layout
def main():
    init_session_state()
    load_css()
    
    with st.sidebar:
        st.title("🌾 PhytoVision AI")
        st.markdown("---")
        
        languages = st.session_state.config['languages']
        lang_options = [lang['code'] for lang in languages]
        lang_format = lambda x: next(l['name'] for l in languages if l['code'] == x)
        
        selected_lang = st.selectbox(
            "🌍 Language / भाषा / Idioma",
            options=lang_options,
            format_func=lang_format,
            index=lang_options.index(st.session_state.language)
        )
        
        if selected_lang != st.session_state.language:
            st.session_state.language = selected_lang
            st.rerun()
            
    st.title("Welcome to PhytoVision AI")
    st.markdown("**Enterprise-grade crop disease diagnosis system.**")
    st.info("👈 Please select a module from the sidebar to begin.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 📸 Quick Analysis
        Upload images of leaves or fruits to detect diseases instantly.
        """)
    with col2:
        st.markdown("""
        ### 🎥 Live Monitoring
        Connect your camera for real-time field monitoring.
        """)

if __name__ == "__main__":
    main()