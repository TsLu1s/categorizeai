import streamlit as st
import home
import importlib

def get_pages():
    # Keep the original PAGES structure with st_pages prefix
    # Define pages and their icons, grouped by use case
    return {
        "Home": {
            "icon": "house",
            "func": "home.run" 
        },
        "Supervised Classification": [ 
            {
                "name": "Experiments Analysis",
                "icon": "diagram-3",
                "func": "st_pages.supervised_classification.data_analysis.run"
            },
            {
                "name": "Model Training",
                "icon": "gear",
                "func": "st_pages.supervised_classification.model_training.run"
            },
            {
                "name": "Real-Time Prediction",
                "icon": "clock",
                "func": "st_pages.supervised_classification.real_time_prediction.run"
            },
            {
                "name": "Batch Prediction",
                "icon": "collection",
                "func": "st_pages.supervised_classification.batch_prediction.run"
            }
        ],
        "Zero-Shot Classification": [ 
            {
                "name": "Batch Prediction",
                "icon": "collection",
                "func": "st_pages.zero_shot.zero_shot_prediction.run"
            },
        ],
        "LLM Text Classification": [  
            {
                "name": "LLM's Management",
                "icon": "cloud-arrow-up",
                "func": "st_pages.llm_classifier.model_management.run"
            },
            {
                "name": "Batch Prediction",
                "icon": "collection",
                "func": "st_pages.llm_classifier.llm_text_classifier.run"
            },
        ]
    }

def load_module_function(func_path: str) -> callable:
    """
    Dynamically imports and returns a function from a module path string.
    
    Args:
        func_path: String path to function (e.g. 'module.submodule.function')
    
    Returns:
        callable: The imported function
    """
    try:
        module_path, func_name = func_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, func_name)
    except ModuleNotFoundError as e:
        st.error(f"Module not found: {module_path}")
        st.error(f"Error details: {str(e)}")
        return home.run
    except AttributeError as e:
        st.error(f"Function '{func_name}' not found in module {module_path}")
        st.error(f"Error details: {str(e)}")
        return home.run

def get_page_function(selected_category: str, selected_option: str = None) -> callable:
    """
    Returns the appropriate page function based on category and option selection.
    
    Args:
        selected_category: The main category selected
        selected_option: Optional sub-option selection
    
    Returns:
        callable: Function to render the selected page
    """
    PAGES = get_pages()

    try:
        if selected_option:
            # Find matching page for category + option
            selected_page = next(
                page for page in PAGES[selected_category]
                if page["name"] == selected_option
            )
            func_path = selected_page["func"]
        else:
            # Handle pages without sub-options
            func_path = PAGES[selected_category]["func"]
            
        return load_module_function(func_path)
        
    except (KeyError, StopIteration) as e:
        st.error(f"Invalid page selection: {selected_category}/{selected_option}")
        st.error(f"Error details: {str(e)}")
        return home.run
    except Exception as e:
        st.error(f"Unexpected error while loading page: {str(e)}")
        return home.run