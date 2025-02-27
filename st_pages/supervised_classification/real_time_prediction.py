import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, Tuple, List
from functools import partial
import scheme.processing as pr

class ModelInference:
    def __init__(
        self,
        input_col: str = 'text',
        target_col: str = 'label',
        preprocessor: Optional[object] = None
    ):
        """
        Initialize the NLP inference interface with flexible experiment handling.
        
        Args:
            input_col: Name of the input text column
            target_col: Name of the target label column
            preprocessor: Optional custom text preprocessor
        """
        self.input_col = input_col
        self.target_col = target_col
        self._preprocessor = preprocessor or pr
        self._lemmatizer = None
        self._stop_words = None
        
        # Will be set when model is loaded
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
            if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
                st.error("Required model files not found. Please ensure the model has been trained.")
                return False
            
            try:
                # Load vectorizer
                self.vectorizer = joblib.load(vectorizer_path)
                
                # Load CatBoost model
                from catboost import CatBoostClassifier
                self.classifier = CatBoostClassifier()
                self.classifier.load_model(model_path)
                
                # Get unique labels
                self.labels = self.classifier.classes_
                self.current_experiment = experiment_name
                
                return True
                
            except Exception as e:
                st.error(f"Error loading model components: {str(e)}")
                return False
            
        except Exception as e:
            st.error(f"Error accessing model files: {str(e)}")
            return False

    def preprocess_and_predict(self, text: str) -> Tuple[pd.Series, pd.DataFrame]:
        """Preprocess text and make prediction."""
        try:
            if not all([self.classifier, self.vectorizer]):
                raise ValueError("Model components not loaded")
            
            # Preprocess text
            preprocess_bound = partial(
                self._preprocessor.preprocess_text,
                lemmatizer=self._lemmatizer,
                stopwords_dict=self._stop_words
            )
            processed_text = preprocess_bound(text)
            
            # Vectorize
            text_vectorized = self.vectorizer.transform([processed_text]).toarray()
            
            # Make predictions
            prediction = self.classifier.predict(text_vectorized)[0]
            probabilities = self.classifier.predict_proba(text_vectorized)
            
            return str(prediction), probabilities
            
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            return "", np.zeros((1, len(self.labels)))

    def preprocess_and_predict(self, text: str) -> Tuple[pd.Series, pd.DataFrame]:
        """Preprocess text and make prediction."""
        try:
            if not all([self.classifier, self.vectorizer]):
                raise ValueError("Model components not loaded")
            
            # Preprocess text
            preprocess_bound = partial(
                self._preprocessor.preprocess_text,
                lemmatizer=self._lemmatizer,
                stopwords_dict=self._stop_words
            )
            processed_text = preprocess_bound(text)
            
            # Vectorize
            text_vectorized = self.vectorizer.transform([processed_text]).toarray()
            
            # Make predictions
            prediction = self.classifier.predict(text_vectorized)[0]
            probabilities = self.classifier.predict_proba(text_vectorized)
            
            return str(prediction), probabilities
            
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            return "", np.zeros((1, len(self.labels)))
    
    def plot_prediction_gauge(self, probabilities: np.ndarray) -> go.Figure:
        """Create gauge chart for prediction probabilities."""
        prob_value = (probabilities[0, 1] * 100 if len(self.labels) == 2 
                     else np.max(probabilities[0]) * 100)

        colors = {
            'low': '#935F4C',     # Red/brown
            'mid': '#F1E2AD',     # Light yellow
            'high': '#935F4C',    # Olive green
            'bar': '#374B5D'      # Navy blue,
            
        }

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confidence Score", 'font': {'size': 14, 'family': 'Inter'}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': colors['bar']},
                'steps': [
                    {'range': [0, 33], 'color': colors['low']},
                    {'range': [33, 66], 'color': colors['mid']},
                    {'range': [66, 100], 'color': colors['high']}
                ],
                'threshold': {
                    'line': {'color': colors['low'], 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))

        fig.update_layout(
            width=200,
            height=350,
            paper_bgcolor='#FFFAE5',
            font={'color': '#1B1821'}
        )
        return fig

    def plot_probability_distribution(self, probabilities: np.ndarray) -> go.Figure:
        """Create bar chart for prediction probabilities across classes."""
        colors = {
            'primary': '#374B5D',
            'secondary': '#935F4C',
            'accent': '#E0D7C7',
            'background': '#FFFAE5',
            'text': '#1B1821'
        }

        # Use consistent colors for up to 5 classes
        bar_colors = [colors['primary'], colors['secondary'], colors['accent'],
                      '#F1E2AD', '#4A6A8A', '#BC7967', '#D4C5B9', '#2C3E50', '#8B4B45'] # Additional colors
        
        fig = go.Figure(data=[
            go.Bar(
                x=self.labels,
                y=probabilities[0],
                marker_color=bar_colors[:len(self.labels)]
            )
        ])

        fig.update_layout(
            height=350,
            paper_bgcolor=colors['background'],
            plot_bgcolor=colors['background'],
            font={'color': colors['text'], 'family': 'Inter'},
            margin=dict(l=40, r=40, t=40, b=40),
            xaxis={'showgrid': False},
            yaxis={
                'title': 'Probability',
                'showgrid': True,
                'gridcolor': colors['accent'],
                'range': [0, 1]
            }
        )
        return fig

    def run(self):
        """Run the inference interface."""
        st.markdown('<h1 class="main-header"> Supervised Classification - Real-Time Prediction</h1>', 
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

        # Load selected model
        if selected_idx is not None:
            experiment_name, dataset_name = experiments[selected_idx]
            if not self.get_saved_model(experiment_name, dataset_name):
                return

            # Input interface
            st.markdown("### Enter Text for Classification")
            input_text = st.text_area(
                "Input text:",
                "",
                height=150,
                help="Enter the text you want to classify"
            )
            col1, col2, col3 = st.columns([1,2,2])
            with col1:
                predict_button = st.button("Predict", type="primary", use_container_width=True)

            if predict_button:
                if not input_text:
                    st.warning("Please enter text to analyze.")
                    return

                try:
                    prediction, probabilities = self.preprocess_and_predict(input_text)
                    
                    if isinstance(probabilities, np.ndarray) and probabilities.size > 0:
                        st.markdown("### Result Analysis")
                                                
                        # Show visualizations
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(self.plot_prediction_gauge(probabilities))
                        with col2:
                            st.plotly_chart(self.plot_probability_distribution(probabilities))

                        # Show prediction first
                        confidence = float(np.max(probabilities[0]))
                        st.info(
                            f"Predicted Class: **{prediction}** (Confidence: {confidence:.2%})",
                            icon="🎯"
                        )

                    else:
                        st.error("Could not generate prediction. Please try again.")
                except Exception as e:
                    st.error(f"Error predicting text: {str(e)}")
def run():
    """Run the appropriate inference configuration based on context."""
    # Initialize and run the appropriate Inference
    inference = ModelInference()
    inference.run()