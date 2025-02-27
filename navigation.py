import streamlit as st
from streamlit_option_menu import option_menu

import sys
import os

import scheme.nav_loading as nav

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Set page config
st.set_page_config(page_title="CategorizeAI", layout="wide")

# Load custom CSS from file
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css('styles.css')
PAGES = nav.get_pages()

# Then use h1 and p with the classes
st.markdown("""
<div class="header">
    <div class="animated-bg"></div>
    <div class="header-content">
        <h1 class="header-title">CategorizeAI</h1>
        <p class="header-subtitle">Multi-Model NLP Text Classification Platform</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation menu
def navigate():

    with st.sidebar:
        st.markdown("""
            <div class="sidebar-header">
                <div class="sidebar-title">CategorizeAI</div>
                <div class="sidebar-subtitle"> Dynamic NLP Platform for Text Classification </div>
            </div>
        """, unsafe_allow_html=True)

        # Custom icons for main categories
        category_icons = {
            "Home": "house",
            "Supervised Classification": "bullseye", 
            "Zero-Shot Classification": "magic",
            "LLM Text Classification": "robot"
        }

        # Main categories
        selected_category = option_menu(
            menu_title="Navigation",
            options=list(PAGES.keys()),
            icons=[category_icons.get(page, "cast") for page in PAGES],
            menu_icon="cast",
            default_index=0
        )

        # Handle sub-options
        if isinstance(PAGES[selected_category], list):
            options = [page["name"] for page in PAGES[selected_category]]
            icons = [page["icon"] for page in PAGES[selected_category]]
            selected_option = option_menu(
                menu_title="Select Option",
                options=options,
                icons=icons,
                menu_icon="list",
                default_index=0
            )
        else:
            selected_option = None
            
        return selected_category, selected_option

# Determine which page to load
selected_category, selected_option = navigate()

# Determine which function to run
page_func = nav.get_page_function(selected_category, selected_option)
page_func()

# Display the footer
st.markdown("""
<div class="footer">
    <p>© 2025 Powered by <a href="https://github.com/TsLu1s" target="_blank">TsLu1s</a> |
         Project Source: <a href="https://github.com/TsLu1s/categorizeai" target="_blank"> CategorizeAI</p>
</div>
""", unsafe_allow_html=True)