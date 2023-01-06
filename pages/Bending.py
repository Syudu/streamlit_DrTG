import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import csv
import streamlit as st
import plotly.express as px 
from st_aggrid import AgGrid
import plotly.graph_objects as go
import time 
import hydralit_components as hc
import base64
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn import metrics
from sklearn.model_selection import train_test_split


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

with open( ".streamlit/style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

st.title("Bending")

st.write(''' The data set used for the ML model: ''')
df = pd.read_csv("Bending_along_plane_eV.csv")
# st.dataframe(df) #Original
df.rename(columns={'Unnamed: 0':'Theta'},inplace=True)
#-----SideBar ----
# x = st.latex(r'\theta')
# y = "Select the values to be displayed:"
st.sidebar.title("Filter here:")
th = df['Theta'].unique()
th = st.sidebar.multiselect(
    "Select the values to be displayed:",
    options= th,
    default = th
)
# x = df.columns.values
df_selection = df.query(
    "Theta == @th"
)

# st.dataframe(df_selection)
AgGrid(df_selection.fillna("").astype("str"), height=338, fit_columns_on_grid_load=True)
# st.title(":bar_chart: Dashboard")
st.markdown('##')

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.write('''### Data description: 

(Note: Description changes based on filtered angles)
''')

df_d = df_selection.describe(include='all')
df_d.insert(0, "Paramters",['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max' ], True)
 
AgGrid(df_d.fillna("").astype("str"), height=257, fit_columns_on_grid_load=True)
# st.write(df.describe(include='all'))

st.write('''
### Visualizing the Data: 

Angle of Symmetric Bending (vs) SPE
''')

fig = go.Figure()
fig.add_trace(
    go.Line(x = df['Theta'], y = df_selection['z1'], name = "Z = 1")
)
fig.add_trace(
    go.Line(x = df['Theta'], y = df_selection['z2'], name = "Z = 2")
)
fig.add_trace(
    go.Line(x = df['Theta'], y = df_selection['z3'], name = "Z = 3")
)
fig.add_trace(
    go.Line(x = df['Theta'], y = df_selection['z4'], name = "Z = 4")
)
fig.add_trace(
    go.Line(x = df['Theta'], y = df_selection['z5'], name = "Z = 5")
)
fig.update_yaxes(title_font=dict(size=12))
fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor='crimson')
fig.update_yaxes(ticks="outside", tickwidth=2, tickcolor='crimson')
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(
    autosize=True,
    width=1400,
    # title={
    #     'text': "",
    #     'font': dict(size = 20),
    #     # 'y':0.5,
    #     # 'x':0.5,
    #     'xanchor': 'center',
    #     'yanchor': 'top'},
    xaxis_title="Angle of Symmetric Bending",
    yaxis_title="SPE",
    yaxis_title_font = dict(size = 20),
    xaxis_title_font = dict(size = 20),
    xaxis_tickfont = dict(size = 15),
    yaxis_tickfont = dict(size = 15),
    legend_title="Legend",
    legend_title_font = dict(size = 14),
    template = 'plotly_dark'
)
st.write(fig)

#Creating a new reshaped dataset

st.write('''
### Creating New Dataset
''')
data = pd.DataFrame()

Z = np.array([])
for x in range (1,6):
    for i in range(11):
        Z=np.append(Z,x)
        
Theta = np.array([])
for i in range(5):
    Theta=np.append(Theta,df['Theta'])
    
eV = np.array([])
for x in range (1,6):
    for i in range(11):
        eV=np.append(eV,df[df.columns[x]][i])
        
data['Z'] = Z
data['Theta'] = Theta
data['eV'] = eV

latex_theta = ''' $ \theta $'''

st.sidebar.subheader(r"Filter here for values of $\theta$ and Z:")
th1 = data['Theta'].unique()
th1 = st.sidebar.multiselect(
    r"Select the values of $\theta$ to be displayed:",
    options= th1,
    default = th1[:11]
)
z1 = data['Z'].unique()
z1 = st.sidebar.multiselect(
    "Select the values of Z to be displayed:",
    options= z1,
    default = z1
)

# x = df.columns.values
data_selection = data.query(
    "(Theta == @th1) and (Z == @z1)"
)

AgGrid(data_selection.fillna("").astype("str"), height=338, fit_columns_on_grid_load=True)

data = pd.DataFrame()

Z = np.array([])
for x in range (1,6):
    for i in range(11):
        Z=np.append(Z,x)
        
Theta = np.array([])
for i in range(5):
    Theta=np.append(Theta,df['Theta'])
    
eV = np.array([])
for x in range (1,6):
    for i in range(11):
        eV=np.append(eV,df[df.columns[x]][i])
        
data['Z'] = Z
data['Theta'] = Theta
data['eV'] = eV
data.head()

X = data[['Z','Theta']]
y = data['eV']

# st.write(type(y))
# y = y.astype(str)
temp_opt = [0.1,0.15,0.2,0.25,0.3] 
st.sidebar.subheader("Choose Polynomial regression model paramters")
s = st.sidebar.select_slider("Choose Test Size (10\% - 30\%):", options = temp_opt, value = 0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(s))

# st.write(X_train)
# X.columns = X.columns.astype(str) 

# poly = make_pipeline(PolynomialFeatures(2), Ridge(alpha=1e-3))
# poly.fit(X_train,y_train)

# predictions = poly.predict(X_test)

# st.write('MAE:', metrics.mean_absolute_error(y_test, predictions))
# st.write('MSE:', metrics.mean_squared_error(y_test, predictions))
# st.write('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

with st.sidebar:
    deg = st.slider('Choose the degree of the polynomial (0 - 12)', min_value = 1.0, max_value = 12.0,step = 1.0, value = 6.0)
    deg = int(deg)

d = "Degree "+ str(deg)
st.write(d)
fig2 = go.Figure()
    # plt.scatter(data['Theta'],data['eV']) #Plots the data

poly = make_pipeline(PolynomialFeatures(deg), Ridge(alpha=1e-3))
poly.fit(X_train,y_train) 
y_pred = poly.predict(X_test)
    # st.write(y_pred)
col1 , col2, col3 = st.columns(3)
with col3: 
        st.write("Predicted Values:")
        st.write(y_pred)
with col2: 
        st.write("Original Values:")
        st.write(y_test)

with col1:
        st.write('Metrics:')
        st.write('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_pred))
        st.write('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_pred))
        st.write('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
fig2.add_trace(
        go.Scatter(x = X_test['Theta'], y = y_pred, name = d+' Prediction', mode="markers", marker=dict(
        color="dark blue"))
        )
fig2.add_trace(
        go.Scatter(x = X_test['Theta'], y = y_test, name = d+ ' Original', mode = 'markers')
    )
fig2.update_xaxes(ticks="outside", tickwidth=2, tickcolor='crimson')
fig2.update_yaxes(ticks="outside", tickwidth=2, tickcolor='crimson')
fig2.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig2.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig2.update_layout(
    autosize=True,
    width=1400,
    # title={
    #     'text': "Angle of Symmetric Bending (vs) SPE",
    #     'y':0.9,
    #     'x':0.5,
    #     'xanchor': 'auto',
    #     'yanchor': 'middle'},
    xaxis_title=r'Angle of Symmetric Bending',
    yaxis_title="SPE",
    legend_title="Legend",
    template = 'plotly_dark',
    xaxis_tickfont = dict(size = 15),
    yaxis_tickfont = dict(size = 15),
    yaxis_title_font = dict(size = 20),
    xaxis_title_font = dict(size = 20)
)

st.write(fig2)
    # fig2.add_trace(
    #     go.scatter(x = X_test['Theta'], y = , name = d)
    # )

        
    
    # st.write(plt)