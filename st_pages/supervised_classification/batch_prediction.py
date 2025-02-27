import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Optional, List, Dict, Tuple
from functools import partial
import scheme.processing as pr
from scheme.plot_analysis import plot_predictions_by_file
class BatchPrediction:
    """Batch prediction interface for processing multiple text files simultaneously."""
    
    def __init__(
        self,
        input_col: str = 'text',
        target_col: str = 'label',
        preprocessor: Optional[object] = None
    ):
        """Initialize the batch prediction interface."""
        self.input_col = input_col
        self.target_col = target_col
        self._preprocessor = preprocessor or pr
        self._lemmatizer = None
        self._stop_words = None
        
        # Model components
        self.classifier = None
        self.vectorizer = None
        self.labels = None
        self.current_experiment = None

    def _get_available_experiments(self) -> List[Tuple[str, str]]:
        """Get list of all available experiments with their trained models."""
        experiments = []
        base_dir = 'st_pages/experiments'
        
        if os.path.exists(base_dir):
            for experiment in os.listdir(base_dir):
                experiment_dir = os.path.join(base_dir, experiment)
                if os.path.isdir(experiment_dir):
                    # Check for model directory
                    model_dir = os.path.join(experiment_dir, 'saved_models')
                    if os.path.exists(model_dir):
                        # Look for model files
                        for model_file in os.listdir(model_dir):
                            if model_file.endswith('_model.cbm'):
                                # Get dataset name from model file
                                dataset_name = model_file.replace('_model.cbm', '')
                                experiments.append((experiment, dataset_name))
        
        return experiments

    def get_saved_model(self, experiment_name: str, dataset_name: str) -> bool:
        """Load saved model and vectorizer for an experiment."""
        try:
            base_dir = f'st_pages/experiments/{experiment_name}'
            model_path = os.path.join(base_dir, 'saved_models', f'{dataset_name}_model.cbm')
            vectorizer_path = os.path.join(base_dir, 'preprocessing', f'{dataset_name}_vectorizer.pkl')
            
            # Verify files exist
            if not os.path.exists(model_path):
                st.error("Model file not found. Please ensure the model has been trained.")
                return False
            if not os.path.exists(vectorizer_path):
                st.error("Vectorizer file not found. Please ensure preprocessing artifacts are available.")
                return False
                
            # Load model components
            from catboost import CatBoostClassifier
            self.classifier = CatBoostClassifier()
            self.classifier.load_model(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.labels = self.classifier.classes_
            self.current_experiment = experiment_name
            
            return True
            
        except Exception as e:
            st.error(f"Error loading model components: {str(e)}")
            return False

    def load_files(self, files: List) -> pd.DataFrame:
        """Load and prepare text files for batch processing."""
        try:
            return pd.DataFrame({
                'filename': [file.name for file in files],
                self.input_col: [file.read().decode('utf-8') for file in files]
            })
        except Exception as e:
            st.error(f"Error loading files: {str(e)}")
            return pd.DataFrame()

    def process_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of texts and generate predictions."""
        try:
        # Preprocess texts
            preprocess_bound = partial(
                self._preprocessor.preprocess_text,
                lemmatizer=self._lemmatizer,
                stopwords_dict=self._stop_words
            )
            
            df['processed_text'] = df[self.input_col].apply(preprocess_bound)
            
            # Vectorize and predict
            X_batch = self.vectorizer.transform(df['processed_text']).toarray()
            predictions = self.classifier.predict(X_batch)
            probabilities = self.classifier.predict_proba(X_batch)
            
            # Flatten predictions to 1D array
            predictions = predictions.flatten()

            # Add predictions and probabilities
            df['prediction'] = predictions
            for i, label in enumerate(self.labels):
                df[f'{label}_probability'] = probabilities[:, i]
            
            return df
            
        except Exception as e:
            st.error(f"Error processing batch: {str(e)}")
            return pd.DataFrame()

    def run(self):
        """Run the batch prediction interface."""
        st.markdown('<h1 class="main-header">Supervised Classification - Batch Prediction</h1>', 
                   unsafe_allow_html=True)
        st.markdown("---")

        # Get available experiments
        experiments = self._get_available_experiments()
        if not experiments:
            st.error("No trained models found. Please first create an experiment and train a model using the Model Training section.")
            return

        # Create formatted options for the selectbox - experiment name first, then dataset name
        experiment_options = [
            f"{exp.replace('_', ' ').title()} - {ds}"
            for exp, ds in experiments
        ]
        
        selected_idx = st.selectbox(
            "Select Experiment",
            range(len(experiments)),
            format_func=lambda i: experiment_options[i]
        )

        if selected_idx is not None:
            experiment_name, dataset_name = experiments[selected_idx]
            if not self.get_saved_model(experiment_name, dataset_name):
                return

            # File Upload Interface
            st.markdown("### Upload Files for Prediction")
            uploaded_files = st.file_uploader(
                "Upload text files for batch prediction:", 
                accept_multiple_files=True, 
                type=["txt"],
                help="Select multiple .txt files for batch prediction"
            )

            if uploaded_files:
                col1, col2 = st.columns([1, 9])
                with col1:
                    process_button = st.button(
                        "Run Classification",
                        type="primary",
                        use_container_width=True
                    )

                if process_button:
                    with st.spinner("Processing batch..."):
                        # Load and process files
                        df = self.load_files(uploaded_files)
                        results_df = self.process_batch(df)
                        
                        if not results_df.empty:
                            # Display Results
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

                            # Download Results
                            st.markdown("### Export Results")
                            csv = results_df.to_csv(index=False)
                            col1, col2, _ = st.columns([1, 2, 1])
                            with col1:
                                st.download_button(
                                    label="Download Predictions",
                                    data=csv,
                                    file_name=f"{dataset_name}_batch_predictions.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
def run():
    """Run the appropriate inference configuration based on context."""
    
    # Initialize and run the appropriate Inference
    batch_predictor = BatchPrediction()
    batch_predictor.run()

