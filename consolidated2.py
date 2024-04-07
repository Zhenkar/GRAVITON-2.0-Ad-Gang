import numpy as np
import pandas as pd
import ast 
from ast import literal_eval
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2, whiten
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import ward, fcluster
import plotly.express as px
import streamlit as st
import itertools



def distance_picking(Loc1, Loc2, y_low, y_high):
    
    x1, y1 = Loc1[0], Loc1[1]
    x2, y2 = Loc2[0], Loc2[1]
    distance_x = abs(x2 - x1)
    if x1 == x2:
        distance_y1 = abs(y2 - y1)
        distance_y2 = distance_y1
    else:
        distance_y1 = (y_high - y1) + (y_high - y2)
        distance_y2 = (y1 - y_low) + (y2 - y_low)
   
    distance_y = min(distance_y1, distance_y2)
   
    distance = distance_x + distance_y
    return int(distance)

def next_location(start_loc, list_locs, y_low, y_high):
    
    list_dist = [distance_picking(start_loc, i, y_low, y_high) for i in list_locs]
    distance_next = min(list_dist)
    index_min = list_dist.index(min(list_dist))
    next_loc = list_locs[index_min] 
    list_locs.remove(next_loc) 
    return list_locs, start_loc, next_loc, distance_next


def centroid(list_in):
    x, y = [p[0] for p in list_in], [p[1] for p in list_in]
    centroid = [round(sum(x) / len(list_in),2), round(sum(y) / len(list_in), 2)]
    return centroid

 
def centroid_mapping(df_multi):

    df_multi['Coord'] = df_multi['Coord'].apply(literal_eval)

    df_group = pd.DataFrame(df_multi.groupby(['OrderNumber'])['Coord'].apply(list)).reset_index()

    df_group['Coord_Centroid'] = df_group['Coord'].apply(centroid)

    list_order, list_coord = list(df_group.OrderNumber.values), list(df_group.Coord_Centroid.values)
    dict_coord = dict(zip(list_order, list_coord))

    df_multi['Coord_Cluster'] = df_multi['OrderNumber'].map(dict_coord).astype(str)
    df_multi['Coord'] = df_multi['Coord'].astype(str)
    return df_multi

def distance_picking_cluster(point1, point2):

    y_low, y_high = 5.5, 50 

    x1, y1 = point1[0], point1[1]

    x2, y2 = point2[0], point2[1]

    distance_x = abs(x2 - x1)

    if x1 == x2:
        distance_y1 = abs(y2 - y1)
        distance_y2 = distance_y1
    else:
        distance_y1 = (y_high - y1) + (y_high - y2)
        distance_y2 = (y1 - y_low) + (y2 - y_low)

    distance_y = min(distance_y1, distance_y2)

    distance = distance_x + distance_y
    return distance


def create_picking_route(origin_loc, list_locs, y_low, y_high):

    wave_distance = 0

    start_loc = origin_loc

    list_chemin = []
    list_chemin.append(start_loc)
    
    while len(list_locs) > 0: 
        list_locs, start_loc, next_loc, distance_next = next_location(start_loc, list_locs, y_low, y_high)

        start_loc = next_loc
        list_chemin.append(start_loc)

        wave_distance = wave_distance + distance_next 

    wave_distance = wave_distance + distance_picking(start_loc, origin_loc, y_low, y_high)
    list_chemin.append(origin_loc)

    return wave_distance, list_chemin


def create_picking_route_cluster(origin_loc, list_locs, y_low, y_high):

    wave_distance = 0

    distance_max = 0

    start_loc = origin_loc

    list_chemin = []
    list_chemin.append(start_loc)
    
    while len(list_locs) > 0: 
        list_locs, start_loc, next_loc, distance_next = next_location(start_loc, list_locs, y_low, y_high)

        start_loc = next_loc
        list_chemin.append(start_loc)

        if distance_next > distance_max:
            distance_max = distance_next

        wave_distance = wave_distance + distance_next 

    wave_distance = wave_distance + distance_picking(start_loc, origin_loc, y_low, y_high)
    list_chemin.append(origin_loc)

    return wave_distance, list_chemin, distance_max


def process_lines(df_orderlines):

    df_nline = pd.DataFrame(df_orderlines.groupby(['OrderNumber'])['SKU'].count())

    list_ord = list(df_nline.index.astype(int).values)
    list_lines = list(df_nline['SKU'].values.astype(int))

    dict_nline = dict(zip(list_ord, list_lines))
    df_orderlines['N_lines'] = df_orderlines['OrderNumber'].map(dict_nline)

    df_mono, df_multi = df_orderlines[df_orderlines['N_lines'] == 1], df_orderlines[df_orderlines['N_lines'] > 1]
    del df_orderlines

    return df_mono, df_multi

def monomult_concat(df_mono, df_multi):

    df_mono['Coord_Cluster'] = df_mono['Coord']

    df_orderlines = pd.concat([df_mono, df_multi])

    waves_number = df_orderlines.WaveID.max() + 1

    return df_orderlines, waves_number


def cluster_locations(list_coord, distance_threshold, dist_method, clust_start):

    if dist_method == 'euclidian':
        Z = ward(pdist(np.stack(list_coord)))
    else:
        Z = ward(pdist(np.stack(list_coord), metric = distance_picking_cluster))

    fclust1 = fcluster(Z, t = distance_threshold, criterion = 'distance')
    return fclust1


def clustering_mapping(df, distance_threshold, dist_method, orders_number, wave_start, clust_start, df_type):

    list_coord, list_OrderNumber, clust_id, df = cluster_wave(df, distance_threshold, 'custom', clust_start, df_type)
    clust_idmax = max(clust_id)

    dict_map, dict_omap, df, Wave_max = lines_mapping_clst(df, list_coord, list_OrderNumber, clust_id, orders_number, wave_start)
    return dict_map, dict_omap, df, Wave_max, clust_idmax


def cluster_wave(df, distance_threshold, dist_method, clust_start, df_type):

    if df_type == 'df_mono':
        df['Coord_Cluster'] = df['Coord'] 

    df_map = pd.DataFrame(df.groupby(['OrderNumber', 'Coord_Cluster'])['SKU'].count()).reset_index()

    list_coord, list_OrderNumber = np.stack(df_map.Coord_Cluster.apply(lambda t: literal_eval(t)).values), df_map.OrderNumber.values

    clust_id = cluster_locations(list_coord, distance_threshold, dist_method, clust_start)
    clust_id = [(i + clust_start) for i in clust_id]

    list_coord = np.stack(list_coord)
    return list_coord, list_OrderNumber, clust_id, df


def lines_mapping(df, orders_number, wave_start):

    list_orders = df.OrderNumber.unique()

    dict_map = dict(zip(list_orders, [i for i in range(1, len(list_orders))]))

    df['OrderID'] = df['OrderNumber'].map(dict_map)

    df['WaveID'] = (df.OrderID%orders_number == 0).shift(1).fillna(0).cumsum() + wave_start

    waves_number = df.WaveID.max() + 1
    return df, waves_number


def lines_mapping_clst(df, list_coord, list_OrderNumber, clust_id, orders_number, wave_start):
    dict_map = dict(zip(list_OrderNumber, clust_id))
    df['ClusterID'] = df['OrderNumber'].map(dict_map)
    df = df.sort_values(['ClusterID','OrderNumber'], ascending = True)
    list_orders = list(df.OrderNumber.unique())
    dict_omap = dict(zip(list_orders, [i for i in range(1, len(list_orders))]))
    df['OrderID'] = df['OrderNumber'].map(dict_omap)
    df['WaveID'] = wave_start + ((df.OrderID%orders_number == 0) | (df.ClusterID.diff() != 0)).shift(1).fillna(0).cumsum() 
    wave_max = df.WaveID.max()
    return dict_map, dict_omap, df, wave_max


def locations_listing(df_orderlines, wave_id):
    df = df_orderlines[df_orderlines.WaveID == wave_id]
    list_coord = list(df['Coord'].apply(lambda t: literal_eval(t)).values)
    list_coord.sort()
    list_coord = list(k for k,_ in itertools.groupby(list_coord))
    n_locs = len(list_coord)
    n_lines = len(df)
    n_pcs = df.PCS.sum()
    return list_coord, n_locs, n_lines, n_pcs


def df_mapping(df_orderlines, orders_number, distance_threshold, mono_method, multi_method):
    df_mono, df_multi = process_lines(df_orderlines)
    wave_start = 0
    clust_start = 0
    if mono_method == 'clustering':		
        df_type = 'df_mono' 	
        dict_map, dict_omap, df_mono, waves_number, clust_idmax = clustering_mapping(df_mono, distance_threshold, 'custom', 
            orders_number, wave_start, clust_start, df_type)
    else: 
        df_mono, waves_number = lines_mapping(df_mono, orders_number, 0)
        clust_idmax = 0 
    wave_start = waves_number
    clust_start = clust_idmax 
    if multi_method == 'clustering':
        df_type = 'df_multi' 	
        df_multi = centroid_mapping(df_multi)
        dict_map, dict_omap, df_multi, waves_number, clust_idmax  = clustering_mapping(df_multi, distance_threshold, 'custom', 
            orders_number, wave_start, clust_start, df_type)
    else:
        df_multi, waves_number = lines_mapping(df_multi, orders_number, wave_start)
    df_orderlines, waves_number = monomult_concat(df_mono, df_multi)
    return df_orderlines, waves_number


def simulation_wave1(y_low, y_high, orders_number, df_orderlines, list_results, distance_threshold, mono_method, multi_method):
    [list_wid, list_dst, list_route, list_ord, list_lines, list_pcs, list_monomult] = [list_results[i] for i in range(len(list_results))]
    distance_route = 0
    origin_loc = [0, y_low] 	
    df_orderlines, waves_number = df_mapping(df_orderlines, orders_number, distance_threshold, mono_method, multi_method)
    for wave_id in range(waves_number):
        list_locs, n_locs, n_lines, n_pcs = locations_listing(df_orderlines, wave_id)
        wave_distance, list_chemin, distance_max = create_picking_route_cluster(origin_loc, list_locs, y_low, y_high)
        distance_route = distance_route + wave_distance
        monomult = mono_method + '-' + multi_method
        list_wid, list_dst, list_route, list_ord, list_lines, list_pcs, list_monomult = append_results(list_wid, list_dst, list_route, list_ord, list_lines, 
        list_pcs, list_monomult, wave_id, wave_distance, list_chemin, orders_number, n_lines, n_pcs, monomult)
    list_results = [list_wid, list_dst, list_route, list_ord, list_lines, list_pcs, list_monomult]
    return list_results, distance_route


def loop_wave(y_low, y_high, df_orderlines, list_results, n1, n2, distance_threshold, mono_method, multi_method):
    list_ordnum, list_dstw = [], []
    lines_number = len(df_orderlines)
    for orders_number in range(n1, n2):
        list_results, distance_route = simulation_wave1(y_low, y_high, orders_number, df_orderlines, list_results,
            distance_threshold, mono_method, multi_method)
        list_ordnum.append(orders_number)
        list_dstw.append(distance_route)
        print("{} orders/wave: {:,} m".format(orders_number, distance_route))
    [list_wid, list_dst, list_route, list_ord, list_lines, list_pcs, list_monomult] = [list_results[i] for i in range(len(list_results))]
    df_results, df_reswave = create_dataframe(list_wid, list_dst, list_route, list_ord, 
        distance_route, list_lines, list_pcs, list_monomult, list_ordnum, list_dstw)
    return list_results, df_reswave


def simulation_cluster(y_low, y_high, df_orderlines, list_results, n1, n2, distance_threshold):
    mono_method, multi_method = 'normal', 'normal'
    list_results, df_reswave1 = loop_wave(y_low, y_high, df_orderlines, list_results, n1, n2, 
        distance_threshold, mono_method, multi_method)
    mono_method, multi_method = 'clustering', 'normal'
    list_results, df_reswave2 = loop_wave(y_low, y_high, df_orderlines, list_results, n1, n2, 
        distance_threshold, mono_method, multi_method)
    mono_method, multi_method = 'clustering', 'clustering'
    list_results, df_reswave3 = loop_wave(y_low, y_high, df_orderlines, list_results, n1, n2, 
        distance_threshold, mono_method, multi_method)
    [list_wid, list_dst, list_route, list_ord, list_lines, list_pcs, list_monomult] = [list_results[i] for i in range(len(list_results))]
    lines_number = len(df_orderlines)
    df_results = pd.DataFrame({'wave_number': list_wid,
                                'distance': list_dst,
                                'chemins': list_route,
                                'order_per_wave': list_ord,
                                'lines': list_lines,
                                'pcs': list_pcs,
                                'mono_multi':list_monomult})
    df_reswave = process_methods(df_reswave1, df_reswave2, df_reswave3, lines_number, distance_threshold)
    return df_reswave, df_results


def create_dataframe(list_wid, list_dst, list_route, list_ord, distance_route, list_lines, list_pcs, list_monomult, list_ordnum, list_dstw):
    df_results = pd.DataFrame({'wave_number': list_wid,
                                'distance': list_dst,
                                'chemin': list_route,
                                'orders_per_wave': list_ord,
                                'lines': list_lines,
                                'pcs': list_pcs,
                                'mono_multi':list_monomult})
    df_reswave = pd.DataFrame({
        'orders_number': list_ordnum,
        'distance': list_dstw 
        })
    return df_results, df_reswave


def append_results(list_wid, list_dst, list_route, list_ord, list_lines, 
		list_pcs, list_monomult, wave_id, wave_distance, list_chemin, orders_number, n_lines, n_pcs, monomult):

	list_wid.append(wave_id)
	list_dst.append(wave_distance)
	list_route.append(list_chemin)
	list_ord.append(orders_number)
	list_lines.append(n_lines)
	list_pcs.append(n_pcs)
	list_monomult.append(monomult)

	return list_wid, list_dst, list_route, list_ord, list_lines, list_pcs, list_monomult


def process_methods(df_reswave1, df_reswave2, df_reswave3, lines_number, distance_threshold):
    df_reswave1.rename(columns={"distance": "distance_method_1"}, inplace = True)
    df_reswave2.rename(columns={"distance": "distance_method_2"}, inplace = True)
    df_reswave3.rename(columns={"distance": "distance_method_3"}, inplace = True)

    df_reswave = df_reswave1.set_index('orders_number')
    df_reswave['distance_method_2'] = df_reswave2.set_index('orders_number')['distance_method_2']
    df_reswave['distance_method_3'] = df_reswave3.set_index('orders_number')['distance_method_3']

    df_reswave.reset_index().plot.bar(x = 'orders_number', y = ['distance_method_1', 'distance_method_2', 'distance_method_3'], 
        figsize=(10, 6), color = ['black', 'red', 'blue'])

    plt.title("Picking Route Distance for {:,} Order lines / {} m distance threshold".format(lines_number, distance_threshold))
    plt.ylabel('Walking Distance (m)')
    plt.xlabel('Orders per Wave (Orders/Wave)')
    plt.savefig("static/out/{}lines_{}m_3m.png".format(lines_number, distance_threshold))
    plt.show()

    return df_reswave


def orderlines_mapping(df_orderlines, orders_number):
	df_orderlines.sort_values(by='DATE', ascending = True, inplace = True)
	list_orders = df_orderlines.OrderNumber.unique()
	dict_map = dict(zip(list_orders, [i for i in range(1, len(list_orders))]))
	df_orderlines['OrderID'] = df_orderlines['OrderNumber'].map(dict_map)
	df_orderlines['WaveID'] = (df_orderlines.OrderID%orders_number == 0).shift(1).fillna(0).cumsum()
	waves_number = df_orderlines.WaveID.max() + 1
	return df_orderlines, waves_number


def locations_listing1(df_orderlines, wave_id):
	df = df_orderlines[df_orderlines.WaveID == wave_id]
	list_locs = list(df['Coord'].apply(lambda t: literal_eval(t)).values)
	list_locs.sort()
	list_locs = list(k for k,_ in itertools.groupby(list_locs))
	n_locs = len(list_locs)
	return list_locs, n_locs


def simulation_wave(y_low, y_high, origin_loc, orders_number, df_orderlines, list_wid, list_dst, list_route, list_ord):
	distance_route = 0 
	df_orderlines, waves_number = orderlines_mapping(df_orderlines, orders_number)
	for wave_id in range(waves_number):
		list_locs, n_locs, n_lines, n_pcs = locations_listing(df_orderlines, wave_id)
		wave_distance, list_chemin = create_picking_route(origin_loc, list_locs, y_low, y_high)
		distance_route = distance_route + wave_distance
		list_wid.append(wave_id)
		list_dst.append(wave_distance)
		list_route.append(list_chemin)
		list_ord.append(orders_number)
	return list_wid, list_dst, list_route, list_ord, distance_route


def simulate_batch(n1, n2, y_low, y_high, origin_loc, orders_number, df_orderlines):
	list_wid, list_dst, list_route, list_ord = [], [], [], []
	for orders_number in range(n1, n2 + 1):
		list_wid, list_dst, list_route, list_ord, distance_route = simulation_wave(y_low, y_high, origin_loc, orders_number, 
		df_orderlines, list_wid, list_dst, list_route, list_ord)
		print("Total distance covered for {} orders/wave: {:,} m".format(orders_number, distance_route))
	df_waves = pd.DataFrame({'wave': list_wid,
				'distance': list_dst,
				'routes': list_route,
				'order_per_wave': list_ord})
	df_results = pd.DataFrame(df_waves.groupby(['order_per_wave'])['distance'].sum())
	df_results.columns = ['distance']
	return df_waves, df_results.reset_index()


def plot_simulation1(df_results, lines_number):
    fig = px.bar(data_frame=df_results,
        width=1200, 
        height=600,
        x='order_per_wave',
        y='distance',
        labels={ 
            'order_per_wave': 'Wave size (Orders/Wave)',
            'distance': 'Total Picking Walking Distance (m)'})
    fig.update_traces(marker_line_width=1, marker_line_color="black")
    st.write(fig)

def plot_simulation2(df_reswave, lines_number, distance_threshold):
    fig = px.bar(data_frame=df_reswave.reset_index(),
        width=1200, 
        height=600,
        x='orders_number',
        y=['distance_method_1', 'distance_method_2', 'distance_method_3'],
        labels={ 
            'orders_number': 'Wave size (Orders/Wave)',
            'distance_method_1': 'NO CLUSTERING APPLIED',
            'distance_method_2': 'CLUSTERING ON SINGLE LINE ORDERS',
            'distance_method_3': 'CLUSTERING ON SINGLE LINE AND CENTROID FOR MULTI LINE'},
        barmode="group")
    fig.update_traces(marker_line_width=1, marker_line_color="black")
    st.write(fig)


