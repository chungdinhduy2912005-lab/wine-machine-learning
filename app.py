import streamlit as st
from streamlit_option_menu import option_menu

from Home import Home
from Statistics import Statistics
from ModelComparison import ModelComparison
from FeatureSelection import FeatureSelection
from AboutMe import AboutMe

st.set_page_config(
    page_title="Wine Recognition & Selection — GK Project",
    page_icon="🌍",
    layout="wide",
)

# all graphs we use custom css not streamlit
theme_plotly = None


# load Style css
def load_custom_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_custom_css()


# menu bar
def sideBar():
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=[
                "Home",
                "Statistics",
                "Model Comparison",
                "Feature Selection",
                "About Me",
            ],
            icons=["house", "eye", "bar-chart", "funnel", "people"],
            menu_icon="cast",
            default_index=0,
        )
    if selected == "Home":
        Home()
    if selected == "Statistics":
        Statistics()
    if selected == "Model Comparison":
        ModelComparison()
    if selected == "Feature Selection":
        FeatureSelection()
    if selected == "About Me":
        AboutMe()


st.sidebar.image("logo.png", caption="")
sideBar()


# theme
hide_st_style = """

<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
</style>
"""
