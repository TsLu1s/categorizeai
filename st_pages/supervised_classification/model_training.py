import streamlit as st
import pandas as pd
import os
import time
from functools import partial
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import plotly.express as px
from typing import Optional, List, Dict, Tuple
from scheme.models import CatBoost_NLPClassifier
import scheme.processing as pr
from scheme.plot_analysis import ModelVisualization

class ModelTraining:
    """Model training interface with flexible experiment handling."""
    
    def __init__(
        self,
        input_col: str = 'text',
        target_col: str = 'label',
        preprocessor: Optional[object] = None,
    ):
        """Initialize the model training interface."""
        self.input_col = input_col
        self.target_col = target_col
        self._preprocessor = preprocessor or pr
        self._lemmatizer = None
        self._stop_words = None
        self.labels = None
        self.color_scheme = None
        self.viz = ModelVisualization()
        
        # Initialize session state variables
        if 'dataset_loaded' not in st.session_state:
            st.session_state.dataset_loaded = False
        if 'current_dataset' not in st.session_state:
            st.session_state.current_dataset = None
        if 'preprocessing_complete' not in st.session_state:
            st.session_state.preprocessing_complete = False

    def _get_available_experiments(self) -> List[Tuple[str, str]]:
        """Get list of all available experiments with their datasets."""
        experiments = []
        base_dir = 'st_pages/experiments'
        
        # Create base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
        
        if os.path.exists(base_dir):
            for experiment in os.listdir(base_dir):
                experiment_dir = os.path.join(base_dir, experiment)
                if os.path.isdir(experiment_dir):
                    for dataset in os.listdir(experiment_dir):
                        if dataset.endswith('.csv'):
                            experiments.append((experiment, dataset[:-4]))
        
        return experiments

    def _setup_directories(self, experiment_name: str):
        """Create required directories for saving models and preprocessing data."""
        base_dir = f'st_pages/experiments/{experiment_name}'
        directories = [
            os.path.join(base_dir, 'preprocessing'),
            os.path.join(base_dir, 'saved_models')
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        return base_dir

    def _detect_and_setup_labels(self, df: pd.DataFrame):
        """Set up labels and color scheme from dataset."""
        unique_labels = sorted(df[self.target_col].unique())
        st.session_state.n_classes = len(unique_labels)
        
        self.labels = {str(label).lower().replace(' ', '_'): str(label) 
                      for label in unique_labels}
        
        # Fixed color scheme for consistency
        default_colors = [
            '#374B5D',  # Primary
            '#935F4C',  # Secondary
            '#E0D7C7',  # Accent
            '#6B7280',  # Additional colors for more classes
            '#9CA3AF'
        ]
        
        self.color_scheme = {
            label: default_colors[i % len(default_colors)]
            for i, label in enumerate(self.labels.values())
        }
        
        # Debug information for label encoding
        if st.session_state.n_classes == 2:
            st.session_state.positive_label = unique_labels[1]
            st.session_state.negative_label = unique_labels[0]

    def load_experiment_data(self, experiment_name: str, dataset_name: str) -> pd.DataFrame:
        """Load experiment data from CSV and setup labels."""
        try:
            file_path = os.path.join('st_pages/experiments', experiment_name, 
                                    f"{dataset_name}.csv")
            df = pd.read_csv(file_path)
            self._detect_and_setup_labels(df)
            st.session_state.dataset_loaded = True
            st.session_state.current_dataset = (experiment_name, dataset_name)
            return df
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            return pd.DataFrame()
    
    def _display_dataset_info(self):
        """Display dataset information."""
        if st.session_state.dataset_loaded and 'current_df' in st.session_state:
            st.write(st.session_state.current_df.iloc[:, 0:2].head(20))

    def _display_split_info(self):
        """Display data split information."""
        if st.session_state.preprocessing_complete:
            st.markdown("#### Data Split Information")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Set Size", f"{st.session_state.train_size:,}")
            with col2:
                st.metric("Test Set Size", f"{st.session_state.test_size:,}")
    
    def preprocess_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """Preprocess text data and prepare for training."""
        # Process text and split data
        preprocess_bound = partial(
            self._preprocessor.preprocess_text,
            lemmatizer=self._lemmatizer,
            stopwords_dict=self._stop_words
        )
        df[f'processed_{self.input_col}'] = df[self.input_col].apply(preprocess_bound)
        
        train_data, test_data = train_test_split(
            df, test_size=test_size, random_state=42,
            stratify=df[self.target_col]
        )

        # Create TF-IDF features
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
        X_train = tfidf.fit_transform(train_data[f'processed_{self.input_col}']).toarray()
        X_test = tfidf.transform(test_data[f'processed_{self.input_col}']).toarray()

        feature_names = tfidf.get_feature_names_out()
        train_tfidf_df = pd.DataFrame(X_train, columns=feature_names)
        test_tfidf_df = pd.DataFrame(X_test, columns=feature_names)
        
        train_tfidf_df[self.target_col] = train_data[self.target_col].values
        test_tfidf_df[self.target_col] = test_data[self.target_col].values

        st.session_state.preprocessing_complete = True
        st.session_state.train_size = len(train_data)
        st.session_state.test_size = len(test_data)

        return {
            'train_df': train_tfidf_df,
            'test_df': test_tfidf_df,
            'tfidf': tfidf,
            'preprocessed': True
        }
    
    def _display_results(self, y_true, y_pred, y_pred_proba, classifier):
        """Display model performance results and visualizations."""
        st.markdown("----")
        # Classification report
        report_fig = self.viz.plot_classification_report(y_true, y_pred)
        st.plotly_chart(report_fig, use_container_width=True)
        
        # Confusion matrix and feature importance
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(self.viz.plot_confusion_matrix(y_true, y_pred))
        
        with col2:           
            feature_importance_fig = self.viz.plot_feature_importance(
                feature_names=st.session_state.tfidf.get_feature_names_out(),
                top_n=20,
                trained_model=classifier.model
            )
            if feature_importance_fig:
                st.plotly_chart(feature_importance_fig)

        # Binary classification metrics
        if st.session_state.n_classes == 2:
            self.viz.plot_binary_classification_metrics(
                test_df=st.session_state.test_df,
                y_pred_proba=y_pred_proba,
                target_col=self.target_col
            )

    def _save_model_artifacts(self, classifier: CatBoost_NLPClassifier, 
                            preprocessed_data: dict, experiment_name: str, 
                            dataset_name: str):
        """Save model and preprocessing artifacts."""
        try:
            base_dir = self._setup_directories(experiment_name)
            
            # Save model with dataset name for identification
            model_path = os.path.join(base_dir, 'saved_models')
            classifier.save_model(os.path.join(model_path, f'{dataset_name}_model.cbm'))
            
            # Save vectorizer
            preprocess_path = os.path.join(base_dir, 'preprocessing')
            joblib.dump(preprocessed_data['tfidf'], 
                       os.path.join(preprocess_path, f'{dataset_name}_vectorizer.pkl'))
            
            #st.success("Model and preprocessing artifacts saved successfully!")
        except Exception as e:
            st.error(f"Error saving model artifacts: {str(e)}")

    def run(self):
        """Run the model training interface."""
        st.markdown('<h1 class="main-header">Supervised Classification - Model Training</h1>', 
                   unsafe_allow_html=True)
        st.markdown("---")

        # Get available experiments
        experiments = self._get_available_experiments()
        if not experiments:
            st.error("No experiments found. Please first create and save an experiment using the Text Analysis interface.")
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

        if st.button("Load Data"):
            with st.spinner("Loading dataset..."):
                experiment_name, dataset_name = experiments[selected_idx]
                df = self.load_experiment_data(experiment_name, dataset_name)
                if not df.empty:
                    st.session_state.current_df = df
                    self._display_dataset_info()
                
        st.markdown("----")

        if 'current_df' in st.session_state:
            col1, _ = st.columns([1, 2])
            with col1:
                test_size = st.slider("Test Set Size", 0.05, 0.35, 0.2, 0.01)
            
            if st.button("Start Preprocessing"):
                with st.spinner("Preprocessing Text..."):
                    preprocessed_data = self.preprocess_data(
                        st.session_state.current_df, 
                        test_size
                    )
                    st.session_state.update(preprocessed_data)
                
            self._display_split_info()

            if ('train_df' in st.session_state and 
                'test_df' in st.session_state and
                st.button("Train Model")):
                with st.spinner("Training model..."):
                    # Initialize and train classifier
                    classifier = CatBoost_NLPClassifier(
                        label_column=self.target_col,
                        verbose=False
                    )
                    classifier.fit(train_data=st.session_state.train_df)
                    
                    # Generate predictions
                    y_pred = classifier.predict(st.session_state.test_df)
                    y_pred_proba = classifier.predict_proba(st.session_state.test_df)
                    
                    # Save model artifacts
                    experiment_name, dataset_name = st.session_state.current_dataset
                    self._save_model_artifacts(
                        classifier, 
                        st.session_state, 
                        experiment_name,
                        dataset_name
                    )
                    
                    # Display results
                    self._display_results(
                        st.session_state.test_df[self.target_col],
                        y_pred,
                        y_pred_proba,
                        classifier
                    )
        else:
            st.info("Please load a dataset to proceed.")
def run():
    """
    Example configurations and runners for different NLP classification tasks.
    """
    # Initialize and run the appropriate trainer
    trainer = ModelTraining()
    trainer.run()

