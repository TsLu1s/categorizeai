import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def display_classification_approaches():
    """Displays the three classification approaches with cards"""
    
    # Create three columns for the cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="approach-card supervised-card">
            <h3 class="card-title">🎯 Supervised Classification</h3>
            <p class="card-subtitle">Train supervised models with your labeled data</p>
            <p>Upload labeled text files and train specialized supervised models that learn from your data patterns.</p>
            <div class="highlight-box">
                <strong>Best For:</strong>
                <ul>
                    <li>Domain-specific classifications</li>
                    <li>High-precision requirements</li>
                    <li>Consistent text patterns</li>
                </ul>
            </div>
            <div class="workflow-card">
                <div class="workflow-step">
                    <div class="step-number">1</div>
                    <div>Upload labeled text files</div>
                </div>
                <div class="workflow-step">
                    <div class="step-number">2</div>
                    <div>Analyze dataset & train model</div>
                </div>
                <div class="workflow-step">
                    <div class="step-number">3</div>
                    <div>Deploy for real-time & batch prediction</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="approach-card zeroshot-card">
            <h3 class="card-title">🔮 Zero-Shot Classification</h3>
            <p class="card-subtitle">Classify without training using pre-trained models</p>
            <p>Leverage Hugging Face pre-trained zero-shot models to classify texts into any categories without specific training.</p>
            <div class="highlight-box">
                <strong>Best For:</strong>
                <ul>
                    <li>Dynamic categorization needs</li>
                    <li>Limited labeled data availability</li>
                    <li>Rapid implementation requirements</li>
                </ul>
            </div>
            <div class="workflow-card">
                <div class="workflow-step">
                    <div class="step-number">1</div>
                    <div>Select a pre-trained model</div>
                </div>
                <div class="workflow-step">
                    <div class="step-number">2</div>
                    <div>Define your classification context & labels</div>
                </div>
                <div class="workflow-step">
                    <div class="step-number">3</div>
                    <div>Upload and classify text files</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="approach-card llm-card">
            <h3 class="card-title">🤖 LLM Text Classification</h3>
            <p class="card-subtitle">Leverage local LLMs for advanced classification</p>
            <p>Use powerful open-source language models through Ollama for context aware reasoning based classification.</p>
            <div class="highlight-box">
                <strong>Best For:</strong>
                <ul>
                    <li>Complex classification contexts</li>
                    <li>Nuanced text understanding</li>
                    <li>Reasoning-based categorization</li>
                </ul>
            </div>
            <div class="workflow-card">
                <div class="workflow-step">
                    <div class="step-number">1</div>
                    <div>Install Ollama & select LLM</div>
                </div>
                <div class="workflow-step">
                    <div class="step-number">2</div>
                    <div>Provide context & labels</div>
                </div>
                <div class="workflow-step">
                    <div class="step-number">3</div>
                    <div>Process and classify files with LLM reasoning</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_homepage():
    """Run the NLP Classification home page interface."""
    
    # === Main Header ===
    st.markdown('<h1 class="main-header">CategorizeAI - Home</h1>', unsafe_allow_html=True)
    st.markdown('---')
    # Create three tabs
    tab1, tab2 = st.tabs(["🏠 Overview", "📚 Documentation"])

    # Tab 1: Overview
    with tab1:
        
        # Display the classification approaches
        display_classification_approaches()
           
        # Use case examples
        st.markdown('---')
        st.markdown("### Use Cases Examples")
        
        use_cases = [
                {
                    "title": "Sentiment Analysis",
                    "description": "Analyze text to determine emotional tone, sentiment polarity, and subjective opinions. Essential for brand monitoring, customer feedback analysis, and review categorization.",
                    "link": "https://www.kaggle.com/datasets/emrebulbul/imdbdataset"
                },
                {
                    "title": "Spam Detection",
                    "description": "Identify and filter unwanted communications by distinguishing between legitimate and unsolicited content. Applied across email systems, comment sections, and messaging platforms.",
                    "link": "https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset"
                },
                {
                    "title": "Text Document Classification",
                    "description": "Categorize documents into predefined classes based on content analysis. Enables automated document routing, information retrieval, and context analysis.",
                    "link": "https://www.kaggle.com/datasets/shivamkushwaha/bbc-full-text-document-classification"
                },
                {
                    "title": "Fake News Detection",
                    "description": "Identify potentially misleading information through linguistic pattern analysis, source credibility assessment, and contextual verification. Critical for information integrity in digital ecosystems.",
                    "link": "https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/data"
                }
            ]
        # Display use cases in two columns
        col1, col2 = st.columns(2)
        
        for i, case in enumerate(use_cases):
            with col1 if i % 2 == 0 else col2:
                st.markdown(f"""
                <div style="padding: 1rem; background-color: white; border-radius: 0.5rem; margin-bottom: 1rem; border-top: 6px solid #935F4C; border-bottom: 6px solid #374B5D;">
                    <h4 style="margin-top: 0;">{case['title']}</h4>
                    <p>{case['description']}</p>
                    <a href="{case['link']}" target="_blank" class="custom-download-button2"> Data Source </a>
                </div>
                """, unsafe_allow_html=True)

    # Tab 3: Documentation
    with tab2:
        # References section
        st.markdown("### 📚 References & Documentation")
        
        st.markdown("""
        **General Resources:**
        - [Streamlit Documentation](https://docs.streamlit.io/)
        - [Catboost Documentation](https://catboost.ai/docs/en/)
        - [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
        - [Hugging Face Zero-Shot Classifiers](https://huggingface.co/models?pipeline_tag=zero-shot-classification&sort=trending)
        
        **Ollama Resources:**
        - [Ollama Model Library](https://ollama.com/library)
        - [Ollama GitHub Repository](https://github.com/ollama/ollama)
        - [Installation Guide](https://ollama.ai/download)
        - [API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
        """)

def run():
    display_homepage()
