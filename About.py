import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import csv
import streamlit as st
import plotly.express as px 
from st_aggrid import AgGrid
import hydralit_components as hc
import time
import cv2
import streamlit.components.v1 as html
from streamlit_option_menu import option_menu
from manim import *
from manimlib.imports import *
import json
from streamlit_lottie import st_lottie as st_l
import base64

st.set_page_config(page_title="Dr. TG's Work",
    page_icon=":bar_chart:", layout="wide")

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

add_bg_from_local('blue_bg_3.png')  

# my_bar = st.progress(0)

# for percent_complete in range(100):
#     time.sleep(0.1)
#     my_bar.progress(percent_complete + 1)
with hc.HyLoader('Loading...',hc.Loaders.pacman):
    time.sleep(1.5)
st.sidebar.image("Dr_TG.png", use_column_width=True)
# with st.sidebar:
#     choose = option_menu("App Gallery", ["About", "Photo Editing", "Project Planning", "Python e-Course", "Contact"],
#                          icons=['house', 'camera fill', 'kanban', 'book','person lines fill'],
#                          menu_icon="app-indicator", default_index=0,
#                          styles={
#         "container": {"padding": "5!important", "background-color": "#fafafa"},
#         "icon": {"color": "orange", "font-size": "25px"}, 
#         "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
#         "nav-link-selected": {"background-color": "#02ab21"},
#     }
#     )

with open( ".streamlit/style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

def load_lottiefile(path: str):
    with open(path) as f:
        data = json.load(f)
    return data

# st.header('This is an experiment')
st.title(":bar_chart: Dashboard for Bending and Rotation")

st.markdown('##')
st.write('''
### Welcome to Dr. Tribikram Gupta's Dashboard on Research & Development. 
To know more about Dr. Tribikram Gupta, [Click here](https://www.rvce.edu.in/ph-faculty-tg).
This page is dedicated towards Rotation & Bending. ''')
st.write(''' 
### Contact Information:
Dr. Tribikram Gupta

Dept. of Physics, RV College of Engineering - 059


E-mail: [tgupta@rvce.edu.in](mailto:tgupta@rvce.edu.in) 
''')

lottie_file = "lottiefiles/ukr8iAGmJf.json"
lottie_ct = load_lottiefile(lottie_file)

# with col2:
    # st.write('hey')
with st.sidebar:
        st_l(
            lottie_ct,
            loop = True,
            # speed = 2.5,
            quality = "low", #medium ; high
            reverse = False, 
            # rendered = "svg" #canvas
            width=400
        )




col1, col2 = st.columns(2)

with col1:
    st.header(":mailbox: Get In Touch With Me!")



    contact_form = """
<form action="https://formsubmit.co/sudarshan.balaji2000@gmail.com" method="POST">
     <input type="hidden" name="_captcha" value="false">
      <input type="hidden" name="_next" value="https://yourdomain.co/About.py">
     <input type="text" name="name" placeholder="Your name" required>
     <input type="email" name="email" placeholder="Your email" required>
     <textarea name="message" placeholder="Your message here"></textarea>
     <button type="submit">Send</button>
</form>
"""

    st.markdown(contact_form, unsafe_allow_html=True)
# my_expander.write('Hello there!')
# clicked = my_expander.button('Click me!')
    


# st.video('MovingFrameBox.mp4')