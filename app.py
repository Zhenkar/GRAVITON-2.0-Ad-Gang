import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from streamlit import util
from consolidated2 import (
    distance_picking,
    next_location,
    create_picking_route,
    orderlines_mapping,
    locations_listing,
    df_mapping,
    simulation_wave,
    simulate_batch,
    loop_wave,
    simulation_cluster,
    create_dataframe,
    process_methods,
    plot_simulation1,
    plot_simulation2
)


st.set_page_config(page_title="Improve Warehouse Productivity using Order Batching",
                   initial_sidebar_state="expanded",
                   layout='wide',
                   page_icon="🛒")


with open('wave.css') as f:
    css = f.read()


st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)



@st.cache(persist=False,
          allow_output_mutation=True,
          suppress_st_warning=True,
          show_spinner=True)

def load(filename, n):
    df_orderlines = pd.read_csv(IN + filename).head(n)
    return df_orderlines


y_low, y_high = 5.5, 50

origin_loc = [0, y_low]

distance_threshold = 35
distance_list = [1] + [i for i in range(5, 100, 5)]
IN = 'input/'

list_wid, list_dst, list_route, list_ord, list_lines, list_pcs, list_monomult = [], [], [], [], [], [], []
list_results = [list_wid, list_dst, list_route, list_ord, list_lines, list_pcs, list_monomult]  # Group in list

list_ordnum, list_dstw = [], []


st.header("SINGLE PICKER ROUTING PROCESS")
st.subheader('''
        SELECT THE SHELF SIZE
    ''')
col1, col2 = st.columns(2)
with col1:
	n = st.slider(
				'THOUSAND VALUES', 1, 200 , value = 5)
with col2:
	lines_number = 1000 * n 
	st.write('''🛠️{:,} \
		order lines'''.format(lines_number))

st.subheader('''
        ENTER THE MINIMUM AND MAXIMUM LIMIT ''')
col_11 , col_22 = st.columns(2)
with col_11:
	n1 = st.slider(
				'MINIMUM LIMIT', 0, 20 , value = 1)
	n2 = st.slider(
				'MAXIMUM LIMIT', n1 + 1, 20 , value = int(np.max([n1+1 , 10])))
with col_22:
		st.write('''[MIN , MAX] = [{:,}, {:,}]'''.format(n1, n2))

start_1= False
if st.checkbox('SIMULATION 1: START CALCULATION',key='show', value=False):
    start_1 = True

if start_1:
	df_orderlines = load('df_lines.csv', lines_number)
	df_waves, df_results = simulate_batch(n1, n2, y_low, y_high, origin_loc, lines_number, df_orderlines)
	plot_simulation1(df_results, lines_number)
      

st.header("**Quantum-Powered Smart Batch Picking in Warehouses using QAOA**")
st.subheader('''
        Select the Shelf Size
    ''')
col1, col2 = st.columns(2)
with col1:
    n_ = st.slider(
        'Thousand Values', 1, 200, value=5)
with col2:
    lines_2 = 1000 * n_
    st.write('''{:,} \
        Shelf Line'''.format(lines_2))

start_2 = False
if st.checkbox('Calculation Commencement', key='show_2', value=False):
    start_2 = True

if start_2:
    df_orderlines = load('df_lines.csv', lines_2)
    df_reswave, df_results = simulation_cluster(y_low, y_high, df_orderlines, list_results, n1, n2,
                                                distance_threshold)
    plot_simulation2(df_reswave, lines_2, distance_threshold)
