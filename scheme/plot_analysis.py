import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from collections import Counter
from wordcloud import WordCloud
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.metrics import classification_report


from typing import Optional, List, Dict, Any

##########################################################################################
############################## Data Analysis Section

class DataAnalyzer:
    def __init__(self):
        self.colors = {
            'primary': '#374B5D',    # Muted Navy
            'secondary': '#935F4C',   # Tan
            'background': '#FFFAE5',  # Ivory
            'text': '#1B1821',       # Black
            'accent': '#E0D7C7',      # Khaki
            'highlight': '#F1E2AD',   # Yellow
            'icon': '#56382D'        # Dark Brown
        }
        
        self.layout_config = {
            'width': 800,
            'height': 600,
            'paper_bgcolor': self.colors['background'],
            'plot_bgcolor': self.colors['background'],
            'margin': dict(l=80, r=80, t=40, b=80),
            'font_family': 'Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue'
        }

    def _create_base_layout(self, title: str = "", subtitle: str = None) -> Dict[str, Any]:
        """Create a standardized base layout for plots without titles."""
        layout = {
            **self.layout_config,
            'shapes': [{
                'type': 'rect',
                'xref': 'paper',
                'yref': 'paper',
                'x0': 0,
                'y0': 0,
                'x1': 1,
                'y1': 1,
                'line': dict(color=self.colors['secondary'], width=2)
            }]
        }
        return layout

    def _create_axis_layout(self, title: str) -> Dict[str, Any]:
        """Create standardized axis layout."""
        return {
            'title': {
                'text': title,
                'font': {
                    'color': self.colors['text'],
                    'size': 14,
                    'family': self.layout_config['font_family']
                }
            },
            'gridcolor': self.colors['accent'],
            'tickfont': dict(
                color=self.colors['text'],
                size=12,
                family=self.layout_config['font_family']
            ),
            'zeroline': True,
            'zerolinecolor': self.colors['secondary'],
            'zerolinewidth': 2
        }

    def plot_label_distribution(self, df: pd.DataFrame, target_col: str, 
                              color_map: Dict[str, str], title: str) -> go.Figure:
        """Create pie chart showing label distribution."""
        label_counts = df[target_col].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=label_counts.index,
            values=label_counts.values,
            hole=0.4,
            marker=dict(
                colors=[
                    self.colors['primary'],
                    self.colors['secondary'],
                    self.colors['icon'],
                    '#F1E2AD', '#4A6A8A', '#BC7967', '#D4C5B9', '#2C3E50', '#8B4B45' # Additional colors
                ],
                line=dict(color=self.colors['background'], width=2)
            ),
            textinfo='label+percent',
            textfont=dict(
                size=14, 
                family=self.layout_config['font_family'],
                color=self.colors['text']
            ),
            hovertemplate="<b>%{label}</b><br>" +
                         "Count: %{value}<br>" +
                         "Percentage: %{percent}<extra></extra>"
        )])
        
        fig.update_layout(self._create_base_layout())
        return fig

    def plot_text_length_distribution(self, df: pd.DataFrame, input_col: str, 
                                    target_col: str, color_map: Dict[str, str], 
                                    title: str) -> go.Figure:
        """Create histogram of text lengths by label."""
        df = df.copy()
        df['text_length'] = df[input_col].str.len()
        
        fig = go.Figure()
        
        colors = [
            self.colors['primary'],
            self.colors['secondary'],
            self.colors['icon'],
            self.colors['highlight'],

        ]
        
        unique_labels = sorted(df[target_col].unique())
        label_colors = {
            label: colors[i % len(colors)]
            for i, label in enumerate(unique_labels)
        }
        
        for label in unique_labels:
            mask = df[target_col] == label
            fig.add_trace(go.Histogram(
                x=df[mask]['text_length'],
                name=label,
                marker=dict(
                    color=label_colors[label],
                    line=dict(color=self.colors['background'], width=1)
                ),
                opacity=0.7,
                hovertemplate="Length: %{x}<br>Count: %{y}<extra></extra>"
            ))
        
        layout = self._create_base_layout()
        layout.update({
            'xaxis': self._create_axis_layout('Text Length'),
            'yaxis': self._create_axis_layout('Count'),
            'barmode': 'overlay',
            'legend': {
                'bgcolor': self.colors['background'],
                'bordercolor': self.colors['secondary'],
                'borderwidth': 1,
                'font': dict(
                    color=self.colors['text'],
                    family=self.layout_config['font_family']
                )
            }
        })
        
        fig.update_layout(layout)
        return fig

    def plot_common_words(self, df: pd.DataFrame, label: str, input_col: str,
                         target_col: str, color_map: Dict[str, str],
                         n: int = 20) -> Optional[go.Figure]:
        """Create bar chart of most common words with consistent coloring."""
        if df[df[target_col] == label].empty:
            return None
            
        text = " ".join(str(text) for text in df[df[target_col] == label][input_col])
        word_freq = Counter(text.split()).most_common(n)
        
        unique_labels = sorted(df[target_col].unique())
        colors = [
            self.colors['primary'],
            self.colors['secondary'],
            self.colors['icon'],
            self.colors['highlight']
        ]
        
        label_colors = {
            label: colors[i % len(colors)]
            for i, label in enumerate(unique_labels)
        }
        
        fig = go.Figure(data=[go.Bar(
            x=[word for word, _ in word_freq],
            y=[count for _, count in word_freq],
            marker=dict(
                color=label_colors.get(label, self.colors['primary']),
                line=dict(color=self.colors['background'], width=1)
            ),
            hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>"
        )])
        
        layout = self._create_base_layout()
        layout.update({
            'xaxis': {
                **self._create_axis_layout('Words'),
                'tickangle': 45
            },
            'yaxis': self._create_axis_layout('Count'),
            'showlegend': False
        })
        
        fig.update_layout(layout)
        return fig

    def generate_wordcloud(self, df: pd.DataFrame, label: str, input_col: str, 
                          target_col: str) -> Optional[plt.Figure]:
        """Generate wordcloud visualization."""
        if df[df[target_col] == label].empty:
            return None
            
        text = " ".join(str(text) for text in df[df[target_col] == label][input_col])
        
        # Match dimensions with common words plot
        wordcloud = WordCloud(
            max_font_size=60,
            max_words=100,
            background_color=self.colors['background'],
            width=800,
            height=482,  # Match height with common words plot
            colormap='RdYlBu',
            margin=5
        ).generate(text)
        
        # Create figure with matched dimensions
        fig, ax = plt.subplots(figsize=(8, 6))  # Match figsize with plotly dimensions
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Adjust margins
        
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        
        fig.patch.set_facecolor(self.colors['background'])
        
        return fig


##########################################################################################
############################## Model Training Section
class ModelVisualization:
    """A class for creating standardized model visualization plots."""
    
    def __init__(self):
        # Color scheme
        self.colors = {
            'primary': '#374B5D',    # Navy
            'secondary': '#935F4C',   # Tan
            'background': '#FFFAE5',  # Ivory
            'text': '#1B1821',       # Black
            'accent': '#E0D7C7',     # Khaki
            'highlight': '#F1E2AD',   # Yellow
            'icon': '#56382D'        # Dark brown
        }
        
        # Standard layout settings
        self.layout_config = {
            'width': 800,
            'height': 600,
            'paper_bgcolor': self.colors['background'],
            'plot_bgcolor': self.colors['background'],
            'margin': dict(l=80, r=80, t=120, b=80),
            'font_family': 'arial'
        }

    def _create_base_layout(self, title: str, subtitle: str = None) -> Dict[str, Any]:
        """Create a standardized base layout for all plots."""
        layout = {
            'title': {
                'text': f'<b>{title}</b>' + (f'<br><sup>{subtitle}</sup>' if subtitle else ''),
                'font': {'color': self.colors['text'], 'size': 28,'family': 'Arial, Helvetica, sans-serif' },
                'x': 0.5,
                'y': 0.95,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            **self.layout_config
        }
        
        # Add standard border
        layout['shapes'] = [{
            'type': 'rect',
            'xref': 'paper',
            'yref': 'paper',
            'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1,
            'line': dict(color=self.colors['secondary'], width=2)
        }]
        
        return layout

    def _create_axis_layout(self, title: str) -> Dict[str, Any]:
        """Create standardized axis layout."""
        return {
            'title': {
                'text': f'<b>{title}</b>',
                'font': {'color': self.colors['text'], 'size': 16}
            },
            'gridcolor': self.colors['accent'],
            'tickfont': dict(color=self.colors['text'], size=14, family=self.layout_config['font_family']),
            'zeroline': True,
            'zerolinecolor': self.colors['secondary'],
            'zerolinewidth': 2
        }

    def plot_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
        """Create enhanced visualization of classification metrics."""
        report = classification_report(y_true, y_pred, digits=4, output_dict=True)
        metrics = report['weighted avg']
        
        # Prepare data
        metric_data = {
            'names': ['Precision', 'Recall', 'F1-Score', 'Accuracy'],
            'values': [
                metrics['precision'],
                metrics['recall'],
                metrics['f1-score'],
                report['accuracy']
            ]
        }
        
        # Create figure
        fig = go.Figure()
        
        # Add bars
        fig.add_trace(go.Bar(
            x=metric_data['names'],
            y=metric_data['values'],
            text=[f'{value:.3f}' for value in metric_data['values']],
            textposition='auto',
            marker=dict(
                color=[self.colors['primary'], self.colors['secondary'], 
                    self.colors['icon'], self.colors['primary']],
                line=dict(color=self.colors['accent'], width=1)
            ),
            hovertemplate='%{x}: %{y:.4f}<extra></extra>'
        ))
        
        # Set layout
        layout = self._create_base_layout('Model Performance Metrics')
        
        # Update axis layouts with enhanced grid configuration
        layout.update({
            'xaxis': self._create_axis_layout('Metrics'),
            'yaxis': {
                **self._create_axis_layout('Score'),
                'range': [0, 1.1],
                'showgrid': True,
                'gridcolor': self.colors['accent'],
                'gridwidth': 1,
                'dtick': 0.2,
                'tickformat': '.1f'
            }
        })
        
        fig.update_layout(layout)
        
        # Add samples annotation
        fig.add_annotation(
            x=0.5,
            y=1.08,
            xref='paper',
            yref='paper',
            text=f'<b>Total Samples:</b> {int(metrics["support"]):,}',
            showarrow=False,
            font=dict(
                size=14,
                color=self.colors['text'],
                family=self.layout_config['font_family']
            ),
            xanchor='center',
            yanchor='bottom'
        )
        
        return fig

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
        """Create enhanced confusion matrix visualization."""
        cm = confusion_matrix(y_true, y_pred)
        unique_labels = sorted(np.unique(y_true))
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=unique_labels,
            y=unique_labels,
            colorscale=[[0, self.colors['accent']], 
                       [0.5, self.colors['secondary']], 
                       [1, self.colors['primary']]],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16, "color": self.colors['text']},
            hoverongaps=False
        ))
        
        layout = self._create_base_layout('Confusion Matrix')
        layout.update({
            'xaxis': {
                **self._create_axis_layout('Predicted Label'),
                'tickangle': 0,
                'showgrid': True,
                'zeroline': False
            },
            'yaxis': {
                **self._create_axis_layout('True Label'),
                'autorange': "reversed",
                'showgrid': True,
                'zeroline': False
            }
        })
        
        fig.update_layout(layout)
        return fig

    def plot_roc_curve(self, y_true: np.ndarray, y_pred: np.ndarray) -> Optional[go.Figure]:
        """Create enhanced ROC curve visualization."""
        unique_labels = sorted(np.unique(y_true))
        if len(unique_labels) != 2:
            return None
        
        positive_label = unique_labels[1]
        y_true_numeric = np.where(y_true == positive_label, 1, 0)
        fpr, tpr, _ = roc_curve(y_true_numeric, y_pred, pos_label=1)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        
        # Add ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, 
            y=tpr,
            name=f'ROC (AUC = {roc_auc:.3f})',
            line=dict(color=self.colors['primary'], width=3),
            fill='tozeroy',
            fillcolor=f'rgba{tuple(list(int(self.colors["primary"][i:i+2], 16) for i in (1, 3, 5)) + [0.1])}'
        ))
        
        # Add random line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash', color=self.colors['secondary'], width=2)
        ))
        
        layout = self._create_base_layout('ROC Curve Analysis', f'Positive class: {positive_label}')
        layout.update({
            'xaxis': self._create_axis_layout('False Positive Rate'),
            'yaxis': self._create_axis_layout('True Positive Rate'),
            'legend': self._create_legend_layout()
        })
        
        fig.update_layout(layout)
        return fig

    def _create_legend_layout(self) -> Dict[str, Any]:
        """Create standardized legend layout."""
        return {
            'font': dict(color=self.colors['text'], size=14, family=self.layout_config['font_family']),
            'bgcolor': f'rgba{tuple(list(int(self.colors["background"][i:i+2], 16) for i in (1, 3, 5)) + [0.8])}',
            'bordercolor': self.colors['secondary'],
            'borderwidth': 1,
            'x': 0.97,
            'y': 0.03,
            'xanchor': 'right',
            'yanchor': 'bottom'
        }

    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred: np.ndarray) -> Optional[go.Figure]:
        """Create enhanced precision-recall curve visualization."""
        unique_labels = sorted(np.unique(y_true))
        if len(unique_labels) != 2:
            return None
            
        positive_label = unique_labels[1]
        y_true_numeric = np.where(y_true == positive_label, 1, 0)
        precision, recall, _ = precision_recall_curve(y_true_numeric, y_pred, pos_label=1)
        avg_precision = average_precision_score(y_true_numeric, y_pred, pos_label=1)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            name=f'PR (AP = {avg_precision:.3f})',
            line=dict(color=self.colors['primary'], width=3),
            fill='tozeroy',
            fillcolor=f'rgba{tuple(list(int(self.colors["primary"][i:i+2], 16) for i in (1, 3, 5)) + [0.1])}'
        ))
        
        layout = self._create_base_layout('Precision-Recall Analysis', f'Positive class: {positive_label}')
        layout.update({
            'xaxis': self._create_axis_layout('Recall'),
            'yaxis': self._create_axis_layout('Precision'),
            'legend': self._create_legend_layout()
        })
        
        fig.update_layout(layout)
        return fig

    def plot_feature_importance(self, feature_names: List[str], top_n: int = 50, trained_model=None) -> Optional[go.Figure]:
        """Create enhanced feature importance visualization."""
        try:
            if trained_model is None:
                print("No trained model provided for feature importance")
                return None
                
            importance = trained_model.get_feature_importance()
            feature_names = self._process_feature_names(feature_names, len(importance))
            
            importance_df = pd.DataFrame({
                'feature': feature_names[:len(importance)],
                'importance': importance
            }).sort_values('importance', ascending=True)
            
            top_features = importance_df.tail(min(top_n, len(importance_df)))
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=top_features['importance'],
                y=top_features['feature'],
                orientation='h',
                marker=dict(
                    color=top_features['importance'],
                    colorscale=[[0, self.colors['secondary']], [1, self.colors['primary']]],
                    line=dict(color=self.colors['accent'], width=1)
                )
            ))
            
            layout = self._create_base_layout('Feature Importance Analysis')
            layout.update({
                'height': max(600, len(top_features) * 25),
                'xaxis': self._create_axis_layout('Importance Score'),
                'yaxis': {
                    **self._create_axis_layout('Features'),
                    'categoryorder': 'total ascending'
                },
                'showlegend': False
            })
            
            fig.update_layout(layout)
            return fig
            
        except Exception as e:
            print(f"Error in feature importance calculation: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _process_feature_names(self, feature_names: List[str], importance_length: int) -> List[str]:
        """Process feature names to match importance length."""
        if len(feature_names) > importance_length:
            return feature_names[:importance_length]
        elif len(feature_names) < importance_length:
            return list(feature_names) + [f'feature_{i}' for i in range(importance_length - len(feature_names))]
        return feature_names
    
    def plot_binary_classification_metrics(self, test_df: pd.DataFrame, y_pred_proba: pd.DataFrame, target_col:str):
        """Plot ROC and PR curves for binary classification."""
        if st.session_state.n_classes != 2:
            return

        # Get unique labels from the data
        unique_labels = sorted(test_df[target_col].unique())
        # Use the second label (positive class) for binary metrics
        positive_label = unique_labels[1]
        
        # Get probabilities for positive class
        if isinstance(y_pred_proba, pd.DataFrame):
            y_pred_proba_binary = y_pred_proba[positive_label].values
        else:
            y_pred_proba_binary = y_pred_proba[:, 1]
        
        col3, col4 = st.columns(2)
        with col3:
            roc_fig = self.plot_roc_curve(
                test_df[target_col],
                y_pred_proba_binary
            )
            if roc_fig:
                st.plotly_chart(roc_fig)
        
        with col4:
            pr_fig = self.plot_precision_recall_curve(
                test_df[target_col],
                y_pred_proba_binary
            )
            if pr_fig:
                st.plotly_chart(pr_fig)

##########################################################################################
############################## Batch Prediction Analysis
def plot_predictions_by_file(df: pd.DataFrame) -> go.Figure:

    # Get probability columns (columns ending with _probability)
    prob_cols = [col for col in df.columns if col.endswith('_probability')]
    
    # Consistent color scheme matching the application theme
    colors = {
        'primary': '#374B5D',    # Primary blue
        'secondary': '#935F4C',   # Secondary brown
        'accent': '#E0D7C7',      # Accent beige
        'background': '#E0D7C7',  # Accent beige
        'text': '#1B1821',        # Text color
        'highlight': '#F1E2AD',   # Highlight yellow
        'icon': '#56382D'         # Dark brown
    }
    
    # Create base color sequence with primary colors first
    color_sequence = [
        colors['primary'], 
        colors['secondary'], 
        colors['accent'],
        colors['icon'],
        colors['highlight']
    ]
    
    # Add more colors if needed
    additional_colors = ['#4A6A8A', '#BC7967', '#D4C5B9', '#2C3E50', '#8B4B45']
    if len(prob_cols) > len(color_sequence):
        color_sequence.extend(additional_colors[:len(prob_cols) - len(color_sequence)])
    
    # Limit to the number of probability columns
    color_sequence = color_sequence[:len(prob_cols)]

    # Create figure
    fig = go.Figure()
    
    # Add vertical separators between files for better visual separation
    for i in range(len(df['filename'])):
        fig.add_vline(
            x=i + 0.5,
            line_width=1,
            line_dash="dash",
            line_color="gray",
            opacity=0.3
        )
    
    # Add prediction bars for each label
    for prob_col, color in zip(prob_cols, color_sequence):
        label = prob_col.replace('_probability', '')
        fig.add_trace(go.Bar(
            name=label,
            x=df['filename'],
            y=df[prob_col],
            marker_color=color,
            marker_line=dict(
                color=colors['background'],
                width=1
            ),
            hovertemplate=
                "<b>%{x}</b><br>" +
                f"{label}: %{{y:.3f}}<br>" +
                "<extra></extra>"
        ))
    
    # Configure layout with consistent styling
    fig.update_layout(
        title={
            'text': 'Prediction Probabilities by File',
            'font': {
                'family': 'Inter, Arial, sans-serif',
                'size': 25,
                'color': colors['text']
            },
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='',
        yaxis_title={
            'text': 'Probability Score',
            'font': {
                'family': 'Inter, Arial, sans-serif',
                'size': 16,
                'color': colors['text']
            }
        },
        barmode='group',
        height=600,
        xaxis=dict(
            range=[-0.5, len(df['filename']) - 0.5],
            rangeslider=dict(
                visible=True,
                thickness=0.05,
                bgcolor=colors['accent'],
                bordercolor=colors['secondary'],
                borderwidth=1
            ),
            tickangle=45,
            tickfont=dict(
                family='Inter, Arial, sans-serif',
                size=12,
                color=colors['text']
            )
        ),
        yaxis=dict(
            range=[0, 1.05],
            gridcolor=colors['accent'],
            gridwidth=1,
            zerolinecolor=colors['secondary'],
            zerolinewidth=2,
            tickfont=dict(
                family='Inter, Arial, sans-serif',
                size=12,
                color=colors['text']
            )
        ),
        margin=dict(b=100, l=50, r=50, t=80),
        plot_bgcolor='white',
        paper_bgcolor=colors['background'],
        font=dict(
            family='Inter, Arial, sans-serif',
            color=colors['text'],
            size=14
        ),
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.15,
            xanchor='center',
            x=0.5,
            bgcolor=colors['background'],
            bordercolor=colors['secondary'],
            borderwidth=1
        ),
        shapes=[{
            'type': 'rect',
            'xref': 'paper',
            'yref': 'paper',
            'x0': 0,
            'y0': 0,
            'x1': 1,
            'y1': 1,
            'line': dict(color=colors['secondary'], width=2)
        }]
    )
    
    return fig