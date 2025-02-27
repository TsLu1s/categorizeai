import streamlit as st
import subprocess

from scheme.ollama_models import (get_model_info, 
                                  get_ollama_models,
                                  create_models_dataframe, 
                                  display_models_library)

def pull_model(model_name):
    try:
        with st.spinner(f'🤖 Downloading {model_name}  ... This may take a bit depending on the model size'):
            # Run the download command
            process = subprocess.run(
                ['ollama', 'pull', model_name],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            if process.returncode == 0:
                st.success(f"✅ Successfully installed {model_name}")
            else:
                st.error(f"❌ Error pulling {model_name}: {process.stderr}")
                
    except Exception as e:
        st.error(f"❌ Error downloading {model_name}: {str(e)}")

def display_models():
    # Main title with styling
    st.markdown(f'<h1 class="main-header"> LLM Text Classification - Models Management</h1>', unsafe_allow_html=True)
    st.markdown('---')

    # Styled download button for Ollama installation
    st.markdown("#### Install Ollama API")
    st.markdown(""" <a href="https://ollama.com/download" target="_blank" class="custom-download-button2">  Download Ollama </a>""", unsafe_allow_html=True)

    st.markdown("#### Install models from the Ollama Library")

    col1, _ = st.columns([3, 6])
    with col1:
        new_model = st.text_input("Enter the name of the model you want to download:", 
                                placeholder="e.g., llama3.2, mistral, gemma... (Press Enter to download)",
                                key="model_input")
        if new_model:  # When Enter is pressed and there's input
            pull_model(new_model)
        # Hardware requirements panel
        st.markdown(
        """
        <div style='padding: 0.8rem; background-color: #374B5D; color: white; border-radius: 0.5rem; height: 220px; display: flex; flex-direction: column;'>
            <h4 style='color: #F1E2AD; margin-top: 0;'>💻 Hardware Requirements</h4>
            <div>
                Minimum RAM requirements by model size:
                <ul style='padding-left: 1.2rem;'>
                <li><strong>1B-7B models:</strong> 8GB RAM</li>
                <li><strong>8B-13B models:</strong> 16GB RAM</li>
                <li><strong>14B-33B models:</strong> 32GB RAM</li>
                <li><strong>34B+ models:</strong> 64GB+ RAM</li>
                </ul>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
        )

    # Local Models Section
    st.markdown("""
    <div class="section-header">
        <h2>Downloaded Language Models</h2>
        <p>Manage your local model collection</p>
    </div>
    """, unsafe_allow_html=True)

    available_models = get_ollama_models()
    if available_models:
        for model in available_models:
            with st.expander(f"📦 {model}"):
                model_info = get_model_info(model)
                if model_info:
                    st.code(model_info, language="yaml")
    else:
        st.info("No models currently installed. Use the Download tab to install models.")

    st.markdown('---') 

    # Apply styling and display
    models_df = create_models_dataframe()
    display_models_library(models_df)
        

def run():
    display_models()
