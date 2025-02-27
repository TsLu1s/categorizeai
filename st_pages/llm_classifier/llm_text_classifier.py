import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from typing import List, Dict

from scheme.plot_analysis import plot_predictions_by_file
from scheme.ollama_models import get_ollama_models
class LLMZeroShotClassifier:
    def __init__(
        self,
        batch_size: int = 10,
        max_inputs: int = 50,
    ):
        """Initialize the LLM zero-shot classification interface."""
        self.batch_size = batch_size
        self.max_inputs = max_inputs
        
        self.available_models = get_ollama_models()
        
        # Will be set when model is loaded
        self.classifier = None
        self.output_parser = None

    def setup_classifier(self, model_name: str, temperature: float = 0.1):
        """Setup the LLM classifier with selected model."""
        try:
            # Initialize the chat model
            chat_model = ChatOllama(
                model=model_name,
                temperature=temperature
            )
            
            # Define response schemas for structured output
            response_schemas = [
                ResponseSchema(name="classification", 
                             description="The most likely label for the text"),
                ResponseSchema(name="probabilities", 
                             description="Dictionary of probabilities for each label")
            ]
            
            # Create output parser
            self.output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
            format_instructions = self.output_parser.get_format_instructions()
            
            # Create prompt template
            template = """You are a text classifier. Analyze the following text within the classifying context and assign probabilities for each of these categories: {labels}
            
            Context for classification:
            {context}

            Text: {text}
            
            Think about this step by step:
            1. Read and understand the text carefully
            2. Consider how well the text matches each possible label
            3. Assign a probability score between 0 and 1 for each label, where:
               - 1.0 means absolute certainty
               - 0.0 means completely inappropriate
               - The sum of all probabilities should be 1.0
            
            {format_instructions}
            """
            
            prompt = ChatPromptTemplate.from_template(template)
            
            # Create classification chain
            self.classifier = LLMChain(
                llm=chat_model,
                prompt=prompt,
                output_parser=self.output_parser
            )
            
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False

    def load_files(self, files) -> pd.DataFrame:
        """Load text files into DataFrame."""
        texts = []
        for file in files:
            if len(texts) >= self.max_inputs:
                st.warning(f"Maximum input limit reached ({self.max_inputs} texts)")
                break
            
            text = file.read().decode('utf-8')
            texts.append({
                'filename': file.name,
                'text': text
            })
        return pd.DataFrame(texts)

    def normalize_probabilities(self, prob_dict: Dict[str, float]) -> Dict[str, float]:
        """Normalize probabilities to ensure they sum to 1.0"""
        total = sum(prob_dict.values())
        if total == 0:
            return {k: 1.0/len(prob_dict) for k in prob_dict}
        return {k: v/total for k, v in prob_dict.items()}

    def process_batch(self, df: pd.DataFrame, labels: List[str], context: str) -> pd.DataFrame:
        """Process batch of texts with zero-shot classification."""
        texts = df['text'].tolist()
        all_results = []
        labels_str = ", ".join(labels)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Update progress
            progress = (i + len(batch_texts)) / len(texts)
            progress_bar.progress(progress)
            #status_text.text(f"Processing batch {i//self.batch_size + 1}...")
            
            # Process each text in batch
            for j, text in enumerate(batch_texts):
                try:
                    result = self.classifier.invoke({
                        "labels": labels_str,
                        "text": text,
                        "context": context,
                        "format_instructions": self.output_parser.get_format_instructions()
                    })
                    
                    # Parse result
                    classification = result['text']['classification']
                    probabilities = result['text']['probabilities']
                    probabilities = self.normalize_probabilities(probabilities)
                    
                except Exception as e:
                    st.error(f"Error processing text: {str(e)}")
                    classification = "error"
                    probabilities = {label: 1.0/len(labels) for label in labels}
                
                # Store results
                result_dict = {
                    'filename': df.iloc[i + j]['filename'],
                    'text': text,
                    'prediction': classification
                }
                
                # Add probability for each label
                for label, score in probabilities.items():
                    result_dict[f'{label}_probability'] = score
                    
                all_results.append(result_dict)
        
        progress_bar.empty()
        status_text.empty()
        
        return pd.DataFrame(all_results)
    def run(self):
        """Run the LLM zero-shot classification interface."""
        st.markdown(f'<h1 class="main-header"> LLM Text Classification - Batch Prediction</h1>', unsafe_allow_html=True)
        st.markdown('---')

        # Model Selection
        st.markdown('### Select Language Model:')

        col1, _ = st.columns([2,8])
        with col1:
            selected_model = st.selectbox(
                '', 
                self.available_models,
                label_visibility='collapsed',
            )
        
        # File Upload
        st.markdown('### Upload Files:')
        files = st.file_uploader(
            'Upload text files for classification:', 
            accept_multiple_files=True, 
            type=['txt'],
            help=f'Select multiple .txt files (max {self.max_inputs})'
        )

        # Labels Configuration
        st.markdown('### Configure Classification:')
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_labels = st.number_input(
                'Number of labels',
                min_value=2,
                max_value=5,
                value=2,
                help='Select number of classification labels'
            )
            
            # Dynamic label inputs
            labels = []
            for i in range(num_labels):
                label = st.text_input(
                    f'Label {i+1}',
                    key=f'label_{i}',
                    placeholder=f'Enter label {i+1}'
                )
                labels.append(label)

        with col2:
            context = st.text_area(
                'Classification Context',
                placeholder='Describe the classification context (e.g., Spam detection for email filtering system)',
                help='Provide context to guide the classification')
            
        # Validate inputs
        inputs_valid = (
            files and len(files) > 0 and
            all(labels) and len(set(labels)) == num_labels and
            context
        )

        # Run Classification
        if st.button('Run Classification', disabled=not inputs_valid):
            if self.setup_classifier(selected_model):
                with st.spinner('Classifying Uploaded Files...'):
                    # Load and process files
                    df = self.load_files(files)
                    results_df = self.process_batch(df, labels, context)
                    
                    # Display results
                    st.markdown('### Classification Results')
                    st.plotly_chart(
                        plot_predictions_by_file(results_df),
                        use_container_width=True
                    )
                    
                    # Summary statistics
                    st.markdown('### Summary Statistics')
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('#### Prediction Distribution')
                        st.dataframe(
                            results_df['prediction'].value_counts().reset_index(),
                            hide_index=True
                        )
                    
                    with col2:
                        st.markdown('#### Average Probabilities')
                        prob_cols = [col for col in results_df.columns if col.endswith('_probability')]
                        avg_probs = results_df[prob_cols].mean().reset_index()
                        avg_probs.columns = ['Label', 'Average Probability']
                        st.dataframe(avg_probs, hide_index=True)
                    
                    # Download results
                    st.markdown('### Download Results')
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        'Download CSV',
                        csv,
                        'classification_results.csv',
                        'text/csv'
                    )
        
        elif not inputs_valid:
            st.info('Please upload files and configure classification settings to proceed.')
        
        # Hardware Requirements Section
        st.markdown("---")
        # Classification Tips 
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
            """
            <div style='padding: 1rem; background-color: #935F4C; color: white; border-radius: 0.5rem; height: 220px; display: flex; flex-direction: column;'>
                <h4 style='color: #F1E2AD; margin-top: 0;'>🎯 Context & Label Selection</h4>
                <div>
                    <ul style='padding-left: 1.2rem;'>
                        <li><strong>Be Specific:</strong> Instead of "sentiment analysis", use "sentiment analysis of customer reviews about product features ... "</li>
                        <li><strong>Choose Distinct Labels:</strong> Use non-overlapping categories for better accuracy</li>
                        <li><strong>Optimal Range:</strong> Use 2-5 labels for best performance</li>
                        <li><strong>Relevance:</strong> Ensure labels directly relate to your data and context</li>
                    </ul>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
            )

        with col2:
            st.markdown(
            """
            <div style='padding: 1rem; background-color: #56382D; color: white; border-radius: 0.5rem; height: 220px; display: flex; flex-direction: column;'>
                <h4 style='color: #F1E2AD; margin-top: 0;'>⚙️ Model & Performance Optimization</h4>
                <div>
                    <ul style='padding-left: 1.2rem;'>
                        <li><strong>Model Size:</strong> Larger models (13B+) generally provide better predictive performance</li>
                        <li><strong>Clean Inputs:</strong> Remove irrelevant text or noise from your data</li>
                        <li><strong>Consistency:</strong> Use uniform formatting across all input texts</li>
                        <li><strong>Resources:</strong> Large batches with bigger models will take longer to process and require more computing resources</li>
                    </ul>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
            )
            
def run():
    classifier = LLMZeroShotClassifier()
    classifier.run()


######## ADD MORE COLORS
######## Make Plot Compatible with 
######## Replace Classification Context Description with detailed instructions