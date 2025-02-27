# CategorizeAI: Multi-Model NLP Text Classification Platform

A versatile NLP text classification application that seamlessly integrates supervised learning, zero-shot classification, and LLM powered classification reasoning within an intuitive interface. Designed for precision and adaptability, it enables sophisticated text categorization across diverse domains and use cases while supporting dynamic experimentation and testing.

## 🌟 Key Features

- **Multi-Model Architecture**: Choose from three distinct classification approaches based on your specific domain, technical needs and data availability.
- **Supervised Classification**: Complete pipeline from dataset creation and analysis to model training and prediction.
- **Zero-Shot Classification**: Classify text into any categories without specific training using state-of-the-art transformer models.
- **LLM-Powered Classification**: Utilize local LLMs through Ollama for context-aware, reasoning-based classification.
- **Experiment Management**: Save, load, and track different classification experiments with customizable settings for reproducible results.
- **Responsive Design**: Modern, responsive UI with animated components and intuitive navigation.

## Streamlit Demo APP

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://categorizeai.streamlit.app/)

## 👏 Acknowledgments

* [Hugging Face](https://huggingface.co/)
* [Ollama](https://ollama.com/)
* [Langchain](https://langchain.com/)
* [Streamlit](https://streamlit.io/)  

## 📋 Prerequisites

- Python 3.10 or higher
- Streamlit
- Transformers
- For LLM Classification: Ollama API (latest version)
- 8GB+ RAM (16GB+ recommended for LLM classification)

## ⚙️ Installation

1. **Clone the Repository**
```bash
git clone https://github.com/TsLu1s/categorizeai.git
cd categorizeai
```

2. **Set Up Conda Environment**

First, ensure you have Conda installed. Then create and activate a new environment with Python 3.10:

```bash
# Create new environment
conda create -n categorizeai_env python=3.10

# Activate the environment
conda activate categorizeai_env
```

3. **Install Dependencies**
   
```bash
pip install -r requirements.txt
```

4. **Install Ollama**
   
Visit Ollama API and follow the installation instructions for your operating system.

<div align="left">
   
[![Download Ollama](https://img.shields.io/badge/DOWNLOAD-OLLAMA-grey?style=for-the-badge&labelColor=black)](https://ollama.com/download)

</div>

5. **Start the Application**

```bash
streamlit run navigation.py
```

## 💻 Usage & Architecture

### 🏠 Home Page

Overview of all classification approaches, use cases, and documentation resources. Designed for easy navigation and quick understanding of application capabilities.

* Homepage dashboard showcasing the three core classification methodologies
* Explore detailed technical specifications and comparative performance capabilities across approaches
* Review comprehensive use case scenarios with external data source samples recommendations
* Navigate through interactive documentation with technical references

### 🎯 Supervised Classification

Train specialized models with your labeled data. Ideal for domain-specific classifications, efficiency requirements, and data with very consistent patterns.

1. **Experiments Analysis**
   * Upload and configure your experiements with specific context and labeled classes
   * Generate comprehensive distributional and statistical analysis
   * Visualize lexical distributions, word frequency patterns, and semantic clustering

2. **Model Training**
   * Select experiment datasets and configure train/test split parameters
   * Process text data with automated preprocessing mechanisms
   * Visualize performance metrics including classification reports and word feature importance

3. **Real-Time Prediction**
   * Select from trained models across different experiments
   * Enter text directly for immediate classification analysis
   * Visualize prediction probability distribution across all potential categories

4. **Batch Prediction**
   * Process and classify multiple text files simultaneously using trained classification models
   * Visualize prediction distributions and confidence levels across files
   * Export detailed prediction results with probability scores for each category

### 🔮 Zero-Shot Classification

Classify text without training using pre-trained transformer models. Perfect for dynamic categorization needs, potentially limited labeled data scenarios, and effective implementation requirements.

1. **Batch Prediction**
   * Select from state-of-the-art transformer-based models optimized for zero-shot classification
   * Configure custom classification labels and hypothesis templates with domain-specific contexts
   * Visualize prediction distributions and confidence levels across files
   * Export comprehensive results with classification metadata

### 🤖 LLM Text Classification

Leverage any of the Ollama LLMs for advanced classification tasks. Best suited for complex classification contexts, nuanced text understanding, and reasoning-based categorization.

1. **LLMs Management**
   * Browse and download Large Language Models through the Ollama API 
   * Manage installed models through an intuitive interface with detailed technical information
   * Install custom models by name with real-time download tracking

2. **Batch Prediction**
   * Select from downloaded LLM to process multiple text files simultaneously
   * Create detailed context prompts with domain-specific instructions for classification purposes
   * Visualize prediction distributions and confidence levels across files
   * Export structured results with probabilities for each category

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See [LICENSE](https://github.com/TsLu1s/categorizeai/blob/main/LICENSE) for more information.

## 🔗 Contact 
 
Luis Santos - [LinkedIn](https://www.linkedin.com/in/lu%C3%ADsfssantos/)
