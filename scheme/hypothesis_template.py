import streamlit as st

class FlexibleHypothesis:
    """Manages flexible hypothesis templates for zero-shot classification."""
    
    def __init__(self):
        # Template patterns with two placeholders: one for context, one for label
        self.templates = [
            "This text is about {context} that is {label}.",                # Original template
            "This contains {context} that could be classified as {label}.", # More natural variant
            "In the context of {context}, this text is {label}.",           # Emphasizes context
            "From a {context} perspective, this text would be {label}.",    # Analytical framing
            "The {context} expressed in this content is {label}.",          # Content-focused
            "This text demonstrates {label} {context}.",                    # Label-first variant
            "When analyzing for {context}, this would be considered {label}.", # Analysis framing
            "This content exhibits {context} that is {label}.",             # Characteristic framing
            "This text represents an example of {label} {context}.",        # Example framing
            "The {context} aspect of this text can be described as {label}." # Aspect-oriented
        ]
    
    def get_template_ui(self, context:str=None, label_example:str=None):
        """Generate UI for template selection with the provided context."""
        with st.expander("Advanced Template Options", expanded=False):
            st.markdown("##### Hypothesis Template Configuration")
            
            # Show templates with the context filled in for preview
            context_preview = context if context else "[context]"
            
            # Format for preview only (replacing both placeholders with visible text)
            formatted_templates = []
            for t in self.templates:
                # Replace {context} with the actual context and {label} with [label] for preview
                preview = t.replace("{context}", f"**{context_preview}**").replace("{label}", "**[label]**")
                formatted_templates.append(preview)
            
            template_index = st.selectbox(
                "Select how the classification prompt is structured:",
                options=range(len(formatted_templates)),
                format_func=lambda i: formatted_templates[i],
                help="Choose a template structure that best fits your classification task"
            )
            
            selected_template = self.templates[template_index]
            
            # Custom template option
            use_custom = st.checkbox("Use custom template")
            final_template = selected_template
            
            if use_custom:
                custom_template = st.text_input(
                    "Custom template",
                    value=selected_template,
                    help="Use {context} for context and {label} for label placeholders"
                )
                final_template = custom_template
                
            # Preview the template with actual context if available
            if context:
                # For preview, replace placeholders with actual values
                preview = final_template.replace("{context}", context).replace("{label}", f"{label_example}")
                st.markdown("##### Preview:")
                st.info(preview)
                
            return final_template
