import pandas as pd
import streamlit as st
import os
import plotly.express as px
from collections import Counter
from functools import partial
from typing import List, Dict, Optional, Any, Tuple
import scheme.processing as pr
from scheme.plot_analysis import DataAnalyzer

class TextAnalysis:
    """
    Text analysis class that supports dynamic label configuration through Streamlit.
    """
    def __init__(self, input_col: str = 'text', target_col: str = 'label'):
        self.analyzer = DataAnalyzer()
        self.input_col = input_col
        self.target_col = target_col
        self._lemmatizer = None
        self._stop_words = None
        self.current_config = None

    def _get_safe_name(self, name: str) -> str:
        """Create a safe directory/file name."""
        return "".join(c if c.isalnum() else "_" for c in name.lower())

    def load_data_from_files(self, uploaded_files: List[Any], label: str) -> pd.DataFrame:
        """Load data from uploaded files into a DataFrame."""
        try:
            data = {
                self.input_col: [],
                self.target_col: []
            }
            
            for file in uploaded_files:
                content = file.read().decode('utf-8')
                if content.strip():
                    data[self.input_col].append(content)
                    data[self.target_col].append(label)
                
            return pd.DataFrame(data)
        except Exception as e:
            st.error(f"Error loading files: {str(e)}")
            return pd.DataFrame()
    def save_dataset(self, df: pd.DataFrame, context: str, experiment_dir: str) -> Optional[str]:
        """Save DataFrame to CSV using context as the base filename."""
        if not context:
            st.warning("Please enter a context for your classification.")
            return None
            
        try:
            safe_context = self._get_safe_name(context)
            i = 0
            while True:
                filename = f"{safe_context}_{i}" if i > 0 else safe_context
                full_path = os.path.join(experiment_dir, f"{filename}.csv")
                if not os.path.exists(full_path):
                    df.to_csv(full_path, index=False)
                    return filename
                i += 1
        except Exception as e:
            st.error(f"Error saving dataset: {str(e)}")
            return None
    def load_saved_dataset(self, filename: str, experiment_dir: str) -> pd.DataFrame:
        """Load a saved dataset from CSV within an experiment directory.
        
        Args:
            filename (str): Name of the dataset file without extension
            experiment_dir (str): Path to the experiment directory
            
        Returns:
            pd.DataFrame: Loaded dataset or empty DataFrame if loading fails
        """
        try:
            full_path = os.path.join(experiment_dir, f"{filename}.csv")
            return pd.read_csv(full_path)
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            return pd.DataFrame()

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

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the data with preprocessing."""
        try:
            df = df.copy()
            preprocess_bound = partial(
                pr.preprocess_text,
                lemmatizer=self._lemmatizer,
                stopwords_dict=self._stop_words
            )
            df[self.input_col] = df[self.input_col].apply(preprocess_bound)
            return df
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            return df

    def analyze_dataset(self, df: pd.DataFrame, context: str, color_scheme: Dict[str, str]) -> None:
        """Perform and display comprehensive dataset analysis."""
        try:
            st.markdown('---')
            
            # Dataset overview
            st.markdown("### Dataset Overview")
            st.write(df.head(15))
            
            # Distribution analysis
            st.markdown(f"### Label Distribution")
            dist_fig = self.analyzer.plot_label_distribution(
                df, self.target_col, color_scheme, context
            )
            st.plotly_chart(dist_fig, use_container_width=True)
            
            # Text length analysis
            st.markdown("### Text Length Analysis")
            length_fig = self.analyzer.plot_text_length_distribution(
                df, self.input_col, self.target_col,
                color_scheme, context
            )
            st.plotly_chart(length_fig, use_container_width=True)
            
            # Word frequency analysis
            st.markdown("### Word Frequency Analysis")
            # Use the original label order from configuration
            present_labels = [label for label in self.current_config['labels'] 
                            if label in df[self.target_col].unique()]
            
            for label in present_labels:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"#### Word Cloud - {label}")
                    wordcloud_fig = self.analyzer.generate_wordcloud(
                        df, label, self.input_col, self.target_col
                    )
                    if wordcloud_fig:
                        st.pyplot(wordcloud_fig)
                
                with col2:
                    st.markdown(f"#### Common Words - {label}")
                    words_fig = self.analyzer.plot_common_words(
                        df, label, self.input_col,
                        self.target_col, color_scheme,
                        20
                    )
                    if words_fig:
                        st.plotly_chart(words_fig, use_container_width=True)
                
                st.markdown("---")
                
        except Exception as e:
            st.error(f"Error in dataset analysis: {str(e)}")

    def _handle_file_upload(self):
        """Handle file upload and processing."""
        st.header("Upload your Files:")
        
        file_uploaders = {}
        uploaded_files_present = []
        
        for label in self.current_config['labels']:
            files = st.file_uploader(
                f"Upload {label} Files",
                accept_multiple_files=True,
                type="txt",
                help=f"Select one or more text files containing {label.lower()} content"
            )
            file_uploaders[label] = files
            if files:
                uploaded_files_present.append(label)

        if len(uploaded_files_present) < 2:
            st.write(f"Please upload files for at least two categories to continue.")
            return

        dfs = []
        for label, files in file_uploaders.items():
            if files:
                df = self.load_data_from_files(files, label)
                if not df.empty:
                    dfs.append(df)

        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

            if st.button("Save and Analyze Experiment"):
                if saved_filename := self.save_dataset(
                    df, 
                    self.current_config['context'],
                    self.current_config['experiment_dir']
                ):
                    with st.spinner('Processing and analyzing experiment...'):
                        #st.success(f"Dataset saved as: {saved_filename}")
                        processed_df = self._process_data(df)
                        self.analyze_dataset(
                            processed_df, 
                            self.current_config['context'], 
                            self.current_config['color_scheme']
                        )

    def _handle_saved_experiments(self):
        """Handle loading and analysis of saved experiments."""
        experiments = self._get_available_experiments()

        if not experiments:
            st.error("No experiments found. Please create a new experiment by selecting 'Upload New Files' and configuring your analysis.")
            return

        # Create formatted options for the selectbox - experiment name first, then dataset name
        experiment_options = [f"{exp.replace('_', ' ').title()} - {ds}" for exp, ds in experiments]
        
        selected_idx = st.selectbox(
            "Select Experiment",
            range(len(experiments)),
            format_func=lambda i: experiment_options[i]
        )

        if st.button("Load Selected Experiment"):
            with st.spinner('Loading experiment...'):
                selected_experiment, selected_dataset = experiments[selected_idx]
                experiment_dir = f'st_pages/experiments/{selected_experiment}'
                df = self.load_saved_dataset(selected_dataset, experiment_dir)
                
                if not df.empty:
                    processed_df = self._process_data(df)
                    default_colors = [
                        self.analyzer.colors['primary'],
                        self.analyzer.colors['secondary'],
                        self.analyzer.colors['accent'],
                        self.analyzer.colors['icon'],
                        self.analyzer.colors['highlight']
                    ]
                    
                    self.current_config = {
                        'experiment_name': selected_experiment.replace('_', ' ').title(),
                        'context': selected_dataset,
                        'labels': df[self.target_col].unique().tolist(),
                        'experiment_dir': experiment_dir,
                        'color_scheme': {
                            label: default_colors[i % len(default_colors)]
                            for i, label in enumerate(df[self.target_col].unique())
                        }
                    }
                    self.analyze_dataset(
                        processed_df, 
                        selected_dataset, 
                        self.current_config['color_scheme']
                    )
    
    def _configure_new_analysis(self):
        """Configure settings for a new analysis."""
        st.markdown("### Configure Experiment:")
        
        col1, _ = st.columns(2)
        with col1:
            experiment_name = st.text_input(
                "Enter experiment name",
                placeholder="e.g., news classification v1",
                key="experiment_name"
            )

            context = st.text_input(
                "Enter context for classification",
                placeholder="e.g., fake news detection, topic classification, sentiment analysis",
                key="context_input"
            )

        col_label, _ = st.columns([1,4])
        with col_label:
            num_labels = st.number_input(
                "Number of labels", 
                min_value=2, 
                max_value=10, 
                value=2,
                help="Select how many labels you want to classify"
            )

        labels = []
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Label Names")
            for i in range(num_labels):
                label = st.text_input(
                    f"Label {i+1}", 
                    key=f"label_{i}",
                    placeholder=f"Enter label {i+1}"
                )
                if label:
                    labels.append(label)

        if not experiment_name or not context or len(labels) != num_labels or len(set(labels)) != num_labels:
            st.warning("Please complete all configuration fields with unique labels.")
            return None

        # Setup paths and color scheme
        experiment_dir = f'st_pages/experiments/{self._get_safe_name(experiment_name)}'
        os.makedirs(experiment_dir, exist_ok=True)

        default_colors = [
            self.analyzer.colors['primary'],
            self.analyzer.colors['secondary'],
            self.analyzer.colors['accent'],
            self.analyzer.colors['icon'],
            self.analyzer.colors['highlight']
        ]
        
        color_scheme = {
            label: default_colors[i % len(default_colors)]
            for i, label in enumerate(labels)
        }

        return {
            'experiment_name': experiment_name,
            'context': context,
            'labels': labels,
            'experiment_dir': experiment_dir,
            'color_scheme': color_scheme
        }
    def run(self):
        """Run the text analysis application."""
        st.markdown('<h1 class="main-header">Supervised Classification - Experiments Analysis</h1>', 
                   unsafe_allow_html=True)
        st.markdown("---")

        # Navigation options first
        st.markdown("### Select Option:")
        option = st.radio(
            label="",
            options=["📤 Upload New Files", "💾 Saved Experiments"],
            label_visibility="collapsed",
            format_func=lambda x: f"### {x}",
            help="Choose whether to upload new files or analyze existing experiments"
        )

        if option == "📤 Upload New Files":
            config = self._configure_new_analysis()
            if config:
                self.current_config = config
                self._handle_file_upload()
        else:
            self._handle_saved_experiments()

def run():
    """Run the sentiment analysis data exploration interface."""
    analyzer = TextAnalysis()
    analyzer.run()


#### Edit title markdowns to a more sophisticated font
#### Change the Select Options with Tabs


