import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Optional, List, Dict
from transformers import pipeline
import json
from scheme.hypothesis_template import FlexibleHypothesis
from scheme.plot_analysis import plot_predictions_by_file


class ZeroShotBatchPrediction:
    def __init__(
        self,
        context: str = 'zero_shot',
        title: str = 'Zero-Shot Classification',
        input_col: str = 'text',
        target_col: str = 'label',
        batch_size: int = 32,
        max_inputs: int = 50,
    ):
        """Initialize the zero-shot batch prediction interface."""
        self.context = context
        self.title = title
        self.input_col = input_col
        self.target_col = target_col
        self.batch_size = batch_size
        self.max_inputs = max_inputs
        self.base_dir = f'st_pages/{context}'
        
        # Load available models
        with open(f'{self.base_dir}/zero_shot_models.json', 'r') as file:
            self.available_models = json.load(file)
            
        # Will be set when model is loaded
        self.classifier = None
        self.labels = None
        self.hypothesis_template = None
        
        # Initialize the hypothesis template manager
        self.template_manager = FlexibleHypothesis()

    def setup_classifier(self, model_key: str, labels: List[str], hypothesis_template: str):
        """Setup the zero-shot classifier with selected model and parameters."""
        try:
            model_info = self.available_models[model_key]
            with st.spinner(f"Loading model {model_key}..."):
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model=model_info['path'],
                    device=-1 # Use -1 for CPU, 0 for GPU
                )
            self.labels = labels
            self.hypothesis_template = hypothesis_template
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False

    def load_files(self, files: List) -> pd.DataFrame:
        """Load text files into DataFrame."""
        texts = []
        for file in files:
            if len(texts) >= self.max_inputs:
                st.warning(f"Maximum input limit reached ({self.max_inputs} texts)")
                break
            text = file.read().decode('utf-8')
            texts.append({
                'filename': file.name,
                self.input_col: text
            })
        return pd.DataFrame(texts)

    def process_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process batch of texts with zero-shot classification."""
        texts = df[self.input_col].tolist()
        all_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Update progress
            progress = (i + len(batch_texts)) / len(texts)
            progress_bar.progress(progress)
            status_text.text(f"Processing batch {i//self.batch_size + 1}...")
            
            batch_results = self.classifier(
                batch_texts,
                self.labels,
                hypothesis_template=self.hypothesis_template,
                multi_label=False
            )
            
            # Process results
            for j, result in enumerate(batch_results):
                idx = i + j
                result_dict = {
                    'filename': df.iloc[idx]['filename'],
                    'text': texts[idx],
                    'prediction': result['labels'][0],
                }
                # Add probability for each label
                for label, score in zip(result['labels'], result['scores']):
                    result_dict[f'{label}_probability'] = score
                all_results.append(result_dict)
        
        progress_bar.empty()
        status_text.empty()
        
        return pd.DataFrame(all_results)

    def run(self):
        """Run the zero-shot batch prediction interface."""
        st.markdown(f'<h1 class="main-header">Zero-Shot Classification - Batch Prediction</h1>', 
                unsafe_allow_html=True)
        
        # Model Selection
        st.markdown('---')
        st.markdown("### Select Pre-trained Model:")
        model_options = list(self.available_models.keys())
        col1, col2 = st.columns([1,3])
        with col1:
            selected_model = st.selectbox(
                "", 
                model_options,
                label_visibility='collapsed'
            )
            
        # Display model source as clickable link
        st.markdown(f""" <a href="{self.available_models[selected_model]['source']}" target="_blank" class="custom-download-button2">  View Model Source </a>""", unsafe_allow_html=True)
        
        # File Upload
        st.markdown(f"### Upload Files for Classification:")
        files = st.file_uploader(
            "Upload text files for batch prediction:", 
            accept_multiple_files=True, 
            type=["txt"],
            help=f"Select multiple .txt files (max {self.max_inputs}) for batch prediction"
        )
        
        # Custom Labels Input
        st.markdown("### Configure Classification:")

        col_label, _ = st.columns([1,9])
        with col_label:
            # Number of labels selector
            num_labels = st.number_input(
                "Number of labels", 
                min_value=2, 
                max_value=10, 
                value=2,
                help="Select how many labels you want to classify"
            )

        # Dynamic label inputs
        labels = []
        col1, col2 = st.columns([1.5,2])
        
        with col1:
            st.markdown("##### Label Names")
            for i in range(num_labels):
                label = st.text_input(
                    f"Label {i+1}", 
                    key=f"label_{i}",
                    placeholder=f"Enter label {i+1}"
                )
                labels.append(label)

        with col2:
            st.markdown("##### Context")
            _context = st.text_input(
                "Enter context for classification",
                placeholder="e.g., fake news detection, topic classification, sentiment analysis, spam detection",
                key="context_input"
            )
        x1, _ = st.columns([1.5,2])
        with x1:
            # Get the template selection UI after context is entered
            template_selection = self.template_manager.get_template_ui(context=_context, label_example=labels[0])
        
        # Validate inputs before enabling run button
        inputs_valid = (
            files is not None and len(files) > 0 and  # Files uploaded
            all(labels) and len(set(labels)) == num_labels and  # Labels valid
            _context is not None and _context.strip() != ""  # Context provided
        )

        # Run Experiment button
        if st.button("Run Classification", disabled=not inputs_valid):
            # Create hypothesis template using selected template and context
            # Replace the placeholders with the appropriate values
            hypothesis = template_selection.replace("{context}", _context).replace("{label}", "{}")

            # Set up classifier
            if not self.setup_classifier(selected_model, labels, hypothesis):
                return

            with st.spinner("Classifying Uploaded Files..."):
                df = self.load_files(files)
                results_df = self.process_batch(df)
                                
                # Display results
                st.plotly_chart(
                    plot_predictions_by_file(results_df),
                    use_container_width=True,
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


                st.markdown("### Download Results")
                # Add hypothesis template to metadata in CSV
                results_df['hypothesis_template'] = hypothesis
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions",
                    data=csv,
                    file_name=f"zeroshot_classification_predictions.csv",
                    mime="text/csv"
                )
        elif not inputs_valid:
            st.info("Please upload files and configure labels to run the experiment.")

        # Classification Tips Section
        st.markdown("---")
        # Classification Tips 
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
            """
            <div style='padding: 1rem; background-color: #935F4C; color: white; border-radius: 0.5rem; height: 180px; display: flex; flex-direction: column;'>
                <h4 style='color: #F1E2AD; margin-top: 0;'>🎯 Context & Hypothesis Template</h4>
                <div>
                    <ul style='padding-left: 1.2rem;'>
                        <li><strong>Task Description:</strong> Clearly define the target labels you're classifying </li>
                        <li><strong>Template Selection:</strong> Choose hypothesis templates that match your specific use case</li>
                        <li><strong>Label Clarity:</strong> Use descriptive, non-overlapping labels that clearly differentiate categories</li>
                    </ul>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
            )

        with col2:
            st.markdown(
            """
            <div style='padding: 1rem; background-color: #56382D; color: white; border-radius: 0.5rem; height: 180px; display: flex; flex-direction: column;'>
                <h4 style='color: #F1E2AD; margin-top: 0;'>⚙️ Zero-Shot Performance Tips</h4>
                <div>
                    <ul style='padding-left: 1.2rem;'>
                        <li><strong>Domain Match:</strong> Analyse pre-trained models properties and suitability to your classification domain</li>
                        <li><strong>Text Length:</strong> Keep input texts concise and focused for more reliable classifications</li>
                        <li><strong>Label Count:</strong> Use 2-5 labels for optimal zero-shot performance</li>
                    </ul>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
            )
            
def run():
    """Run the zero-shot configuration."""
    config = {'batch_size': 32,
              'max_inputs': 1000}
    
    batch_predictor = ZeroShotBatchPrediction(**config)
    batch_predictor.run()

## After loading data, add data preview 
## Test all ZeroShot Models
## Add the Range of Classifications in the plot analysis
## Change CSV to XLSX