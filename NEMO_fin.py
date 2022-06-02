
###########################################################################################################
######## author = Jithin Raghavan
######## website = https://www.linkedin.com/in/jithinraghavan/
######## layout inspired by https://share.streamlit.io/tylerjrichards/streamlit_goodreads_app/books.py
###########################################################################################################

import streamlit as st
import numpy as np
import pandas as pd
import datetime
import pickle
import time
import altair as alt
import matplotlib as mpl
from matplotlib import pyplot as plt
from  matplotlib.ticker import FuncFormatter
import seaborn as sns
import pydeck as pdk
from pydeck.types import String

from chart_studio import plotly
import plotly.subplots
import plotly.graph_objs as go
from plotly.subplots import make_subplots


st.set_page_config(layout="wide")

### Data Import ###
df_database = pd.read_csv("./data/input_data.csv")
# xlsx = pd.ExcelFile('./data/MechanicalMax_Results.xlsx')
summary_df = pd.read_csv("./data/summary.csv")
wkly_ov_df = pd.read_csv("./data/wkly_overall.csv")
dly_ov_df = pd.read_csv("./data/dly_overall.csv")
dly_loc_df = pd.read_csv("./data/dly_loc.csv")

# types = ["Mean","Median","Maximum","Minimum","Sum"]

label_attr_abs_dict = {"Total Orders":"total_orders","Total Multiline Orders":"total_multiline_orders","Total Singleline Orders":"total_singleline_orders","Total Singleline Multiunit Orders":"total_singleline_multiunit_orders","Total Split Orders":"total_split_orders","Total Multiline Split Orders":"total_multiline_split_orders","Total Packages":"total_packages","Total Excess Packages":"total_excess_packages","Lines Per Order":"lines_per_order","Packages Per Order":"packages_per_order","Units Per Order":"units_per_order","Units Per Package":"units_per_package","Total Lines":"total_lines","Total Units":"total_units","Total FC DVS Split Orders":"total_fc_dvs_split_orders","Total FC Store Split Orders":"total_fc_store_split_orders","Total Store DVS Split Orders":"total_store_dvs_split_orders","Total Store Store Split Orders":"total_store_store_split_orders","Total FC Store DVS Split Orders":"total_fc_store_dvs_split_orders","Total FC FC Split Orders":"total_fc_fc_split_orders","Total DVS DVS Split Orders":"total_dvs_dvs_split_orders","Total Single Node Split Orders":"total_single_node_split_orders","Total No Node All Orders":"total_no_node_all_orders","FC Packages":"fc_packages","Store Packages":"str_packages","DVS Packages":"dvs_packages","Average Package Weight":"average_package_weight","FC Average Weight":"fc_average_weight","Str Average Weight":"str_average_weight","DVS Average Weight":"dvs_average_weight","Overall Average Zone":"overall_average_zone","FC Average Zone":"fc_average_zone","Str Average Zone":"str_average_zone","DVS Average Zone":"dvs_average_zone","Average Shipcost":"average_shipcost","FC Average Shipcost":"fc_average_shipcost","Str Average Shipcost":"str_average_shipcost","DVS Average Shipcost":"dvs_average_shipcost","Total Package Weight":"total_package_weight","FC Package Weight":"fc_package_weight","Store Package Weight":"str_package_weight","DVS Package Weight":"dvs_package_weight","Total Package Zone":"total_package_zone","FC Package Zone":"fc_package_zone","Store Package Zone":"str_package_zone","DVS Package Zone":"dvs_package_zone","Total Shipcost":"total_shipcost","FC Shipcost":"fc_shipcost","Str Shipcost":"str_shipcost","DVS Shipcost":"dvs_shipcost","FC Units":"fc_units","Store Units":"str_units","DVS Units":"dvs_units","Packages Promise Missed":"pkg_edd_miss"}

label_attr_p_dict = {"%Multiline Orders":"p_multiline_orders","%Multiline Orders Split":"p_multiline_orders_split","%Orders Split":"p_orders_split","%Singleline Multi Unit Orders":"p_singleline_multi_unit_orders","%FC-DVS Split Orders":"p_fc_dvs_split_orders","%FC-Store Split Orders":"p_fc_store_split_orders","%Store-DVS Split Orders":"p_store_dvs_split_orders","%Store-Store Split Orders":"p_store_store_split_orders","%FC-Store-DVS Split Orders":"p_fc_store_dvs_split_orders","%FC-FC Split Orders":"p_fc_fc_split_orders","%DVS-DVS Split Orders":"p_dvs_dvs_split_orders","%Single Node Split Orders":"p_single_node_split_orders","%No Node All Orders":"p_no_node_all_orders","%Packages Promise Missed":"p_pkg_edd_miss","%Change in Shipcost":"p_delta_shipcost"}

# color_dict = {'IL': '#fc4744', 'CA':'#8c0303', 'MO':'#edd134', 'TX':'#fa2323', 'NM':'#cf0c0c', 'OH':'#e62222','VA':'#1f9900', 'MD':'#fff830', 'MI':'#dbca12', 'NJ':'#d10606', 'NY':'#007512', 'TN':'#b50300','IN':'#1c2afc', 'WA':'#eb3838', 'ID':'#061fc2', 'KY':'#127a18', 'PA':'#005ac2', 'DE':'#0707a8','CO':'#d1332e', 'NH':'#0546b5', 'MA':'#265ade', 'RI':'#2b82d9', 'MT':'#f57171','NV':'#38d433','CT':'#10a30b','WV': '#fc4744', 'ME':'#8c0303', 'LA':'#edd134', 'OK':'#fa2323', 'OR':'#cf0c0c','AR':'#e62222', 'KS':'#1f9900', 'MS':'#fff830', 'DC':'#dbca12', 'AK':'#d10606', 'HI':'#007512','VT':'#b50300', 'UT':'#1c2afc', 'FL':'#eb3838', 'AL':'#061fc2', 'MN':'#127a18', 'GA':'#005ac2','SC':'#0707a8', 'AZ':'#d1332e', 'NC':'#0546b5', 'WI':'#265ade', 'IA':'#2b82d9', 'NE':'#f57171','SD':'#38d433', 'ND':'#10a30b','WY': '#fc4744'}

### Helper Methods ###
def get_unique_scenarios(df_data):
    #Returns unique date range for labels
    return np.unique(df_data.scenario).tolist()

def get_unique_dates(df_data):
    #returns minimum and maximum
    return np.unique(df_data.order_date).tolist()

def get_unique_states(df_data):
    # states = list(np.unique(np.array(df_data.state)))
    # states = [x for x in states if str(x) != 'nan']
    # unique_states = [int(x) for x in states]
    # return unique_states
    return df_data.state.unique().tolist()

def filter_scenario(df_data):
    df_filtered_scenario = pd.DataFrame()
    # scenario_list = selected_scenario.split()
    df_filtered_scenario = df_data[df_data['scenario'].isin([selected_scenario])]
    return df_filtered_scenario

# def filter_date(df_data):
#     df_filtered_date = pd.DataFrame()
#     dates = np.unique(df_data.order_date).tolist()
#     start_raw = start_date.replace("‏‏‎ ‎‏‏‎ ‎","")
#     end_raw = end_date.replace("‏‏‎ ‎‏‏‎ ‎","")
#     start_index = dates.index(start_raw)
#     end_index = dates.index(end_raw)+1
#     dates_selected = dates[start_index:end_index]
#     df_filtered_date = df_data[df_data['order_date'].isin(dates_selected)]
#     return df_filtered_date

def filter_states(df_data):
    df_filtered_state = pd.DataFrame()
    if all_states_selected == 'Select states manually (choose below)':
        df_filtered_state = df_data[df_data['state'].isin(selected_states)]
        return df_filtered_state
    return df_data

def group_measure_by_attribute(aspect,attribute,measure):
    df_data = df_data_filtered
    df_return = pd.DataFrame()

    if(measure == "Mean"):
        df_return = df_data.groupby([aspect]).mean()

    if(measure == "Median"):
        df_return = df_data.groupby([aspect]).median()

    if(measure == "Minimum"):
        df_return = df_data.groupby([aspect]).min()

    if(measure == "Maximum"):
        df_return = df_data.groupby([aspect]).max()

    if(measure == "Sum"):
        df_return = df_data.groupby([aspect]).sum()

    df_return["aspect"] = df_return.index
    if aspect == "state":
        df_return = df_return.sort_values(by=[attribute], ascending = False)
    return df_return

########################
### ANALYSIS METHODS ###
########################

def plot_x_per_scenario(attr,measure):
    rc = {'figure.figsize':(8,4.5),
          'axes.facecolor':'#ffffff',
          'axes.edgecolor': '#ffffff',
          'axes.labelcolor': 'black',
          'figure.facecolor': '#ffffff',
          'patch.edgecolor': '#ffffff',
          'text.color': 'black',
          'xtick.color': 'black',
          'ytick.color': 'black',
          'grid.color': 'grey',
          'font.size' : 8,
          'axes.labelsize': 9,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8}
    plt.rcParams.update(rc)
    fig, ax = plt.subplots()
    ### Goals
    attribute = label_attr_dict[attr]
    df_plot = pd.DataFrame()
    df_plot = group_measure_by_attribute("scenario",attribute,measure)
    ax = sns.barplot(x="aspect", y=attribute, data=df_plot, color = "#b80606")
    y_str = measure + " " + attr + " " + " per Scenario"

    if measure == "Minimum" or measure == "Maximum":
        y_str = measure + " " + attr + " in the Scenario"

    ax.set(xlabel = "Scenario Name", ylabel = y_str)
    if measure == "Mean" or attribute in ["average_zone","average_lines","average_units","average_shipcost","average_package_weight"]:
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'),
                  (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center',
                   va = 'center',
                   xytext = (0, 15),
                   textcoords = 'offset points')
    else:
        for p in ax.patches:
            ax.annotate(format(str(int(p.get_height()))),
                  (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center',
                   va = 'center',
                   xytext = (0, 15),
                   textcoords = 'offset points')
    st.pyplot(fig)


def plot_x_per_state(attr,measure): #total #against, #conceived
    rc = {'figure.figsize':(8,4.5),
          'axes.facecolor':'#ffffff',
          'axes.edgecolor': '#ffffff',
          'axes.labelcolor': 'black',
          'figure.facecolor': '#ffffff',
          'patch.edgecolor': '#ffffff',
          'text.color': 'black',
          'xtick.color': 'black',
          'ytick.color': 'black',
          'grid.color': 'grey',
          'font.size' : 7,
          'axes.labelsize': 10,
          'xtick.labelsize': 7,
          'ytick.labelsize': 7}

    plt.rcParams.update(rc)
    fig, ax = plt.subplots()
    ### Goals
    attribute = label_attr_dict[attr]
    df_plot = pd.DataFrame()
    df_plot = group_measure_by_attribute("state",attribute,measure)
    if specific_state_colors:
        ax = sns.barplot(x="aspect", y=attribute, data=df_plot.reset_index(), palette = color_dict)
    else:
        ax = sns.barplot(x="aspect", y=attribute, data=df_plot.reset_index(), color = "#b80606")
    y_str = measure + " " + attr + " " + "per State"

    if measure == "Minimum" or measure == "Maximum":
        y_str = measure + " " + attr + "in a State"
    ax.set(xlabel = "State Code", ylabel = y_str)
    plt.xticks(rotation=66,horizontalalignment="right")

    if measure == "Mean" or attribute in ["average_zone","average_lines","average_units","average_shipcost","average_package_weight"]:
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'),
                  (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center',
                   va = 'center',
                   xytext = (0, 18),
                   rotation = 90,
                   textcoords = 'offset points')
    else:
        for p in ax.patches:
            ax.annotate(format(str(int(p.get_height()))),
                  (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center',
                   va = 'center',
                   xytext = (0, 18),
                   rotation = 90,
                   textcoords = 'offset points')
    st.pyplot(fig)

def plt_attribute_correlation(aspect1, aspect2):
    df_plot = df_data_filtered
    rc = {'figure.figsize':(8,4.5),
          'axes.facecolor':'#ffffff',
          'axes.edgecolor': '#ffffff',
          'axes.labelcolor': 'black',
          'figure.facecolor': '#ffffff',
          'patch.edgecolor': '#ffffff',
          'text.color': 'black',
          'xtick.color': 'black',
          'ytick.color': 'black',
          'grid.color': 'grey',
          'font.size' : 7,
          'axes.labelsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8}
    plt.rcParams.update(rc)
    fig, ax = plt.subplots()
    asp1 = label_attr_dict[aspect1]
    asp2 = label_attr_dict[aspect2]
    if(corr_type=="Regression Plot (Recommended)"):
        ax = sns.regplot(x=asp1, y=asp2, x_jitter=.1, data=df_plot, color = '#f21111',scatter_kws={"color": "#f21111"},line_kws={"color": "#c2dbfc"})
    if(corr_type=="Standard Scatter Plot"):
        ax = sns.scatterplot(x=asp1, y=asp2, data=df_plot, color = '#f21111')
    if(corr_type=="Violin Plot (High Computation)"):
       ax = sns.violinplot(x=asp1, y=asp2, data=df_plot, color = '#f21111')
    ax.set(xlabel = aspect1, ylabel = aspect2)
    st.pyplot(fig, ax)


# Converting DF into map ready form
def map_ready_data(df_data, metric):
    df = df_data.groupby(['lat','lng']).mean()[['default_throughput','mechmax_throughput']].reset_index()
    df['default_tp_scaled'] = round(df['default_throughput']/10,0).astype(int)
    df['mechmax_tp_scaled'] = round(df['mechmax_throughput']/10,0).astype(int)
    df = df[['lng','lat',metric]]
    map_ready_df = df.reindex(df.index.repeat(df[metric]))
    map_ready_df = map_ready_df.drop(columns =[metric]).reset_index(drop=True)
    return map_ready_df

# Plotting the PyDeck Chart
def plot_map_throughput(data,metric):
    st.write(
        pdk.Deck(
                    map_style="mapbox://styles/mapbox/light-v9",
                    initial_view_state = pdk.ViewState(
                                                        longitude=-95.5556199,
                                                        latitude=39.8097343,
                                                        zoom=2.5,
                                                        pitch=40.5,
                                                        bearing=-27.36,
                                        ),
                    layers=[
                        pdk.Layer(
                                    "HexagonLayer",
                                    map_ready_data(data,metric),
                                    get_position=["lng", "lat"],
                                    auto_highlight=True,
                                    elevation_scale=5000,
                                    pickable=True,
                                    radius = 50000,
                                    elevation_range=[0, 300],
                                    extruded=True,
                                    coverage=1,
                                )
                            ],
                    tooltip={'html': '<b>Avg. %Throughput</b> (scaled 500:1): {elevationValue}',
                            'style': {"backgroundColor": "steelblue","color": "white"}}
                ),
            )

def node_delta_units(data,node_type):
    df = data.groupby(['state','node_type']).sum()[['mechmax_delta_units','uncap_delta_units']].reset_index()
    df_node = df[df['node_type'].isin([node_type])][['state','mechmax_delta_units','uncap_delta_units']]
    df_node_fin = filter_states(df_node)
    return df_node

def plot_delta_units(data, node_type):
    fig = plotly.subplots.make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_yaxes=True, horizontal_spacing=0)

    # BAR PLOT FOR DCs
    # ----------------
    if node_type == 'DC':
        fig.append_trace(go.Bar(x=node_delta_units(data,node_type).mechmax_delta_units, y=node_delta_units(data,node_type).state, orientation='h', showlegend=True,
                                text=node_delta_units(data,node_type).mechmax_delta_units, name='MECHANICAL MAX CAPACITY', marker_color='#595959'), 1, 1)
        fig.append_trace(go.Bar(x=node_delta_units(data,node_type).uncap_delta_units, y=node_delta_units(data,node_type).state, orientation='h', showlegend=True,
                                text=node_delta_units(data,node_type).uncap_delta_units, name='UNCAPACITATED', marker_color='#b20710'), 1, 2)
    elif node_type == 'STR':
        fig.append_trace(go.Bar(x=node_delta_units(data,node_type).mechmax_delta_units, y=node_delta_units(data,node_type).state, orientation='h', showlegend=True,
                                text=node_delta_units(data,node_type).mechmax_delta_units, name='MECHANICAL MAX CAPACITY', marker_color='#ff9900'), 1, 1)
        fig.append_trace(go.Bar(x=node_delta_units(data,'STR').uncap_delta_units, y=node_delta_units(data,'STR').state, orientation='h', showlegend=True,
                                text=node_delta_units(data,'STR').uncap_delta_units, name='UNCAPACITATED', marker_color='#000099'), 1, 2)

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, categoryorder='total descending', ticksuffix=' ', showline=False)
    fig.update_traces(hovertemplate=None)

    fig.update_layout(hovermode="x",
                      height=500,
                      template='ggplot2',
                      xaxis_title='UNITS',
                      yaxis_title="STATES",
                      plot_bgcolor='white',
                      paper_bgcolor='white',
                      title_font=dict(size=20, color='#221f1f',family="Lato, sans-serif"),
                      font=dict(color='#221f1f'),
                      legend=dict(orientation="h", yanchor="bottom",y=1,xanchor="left"),
                      hoverlabel=dict(bgcolor="black", font_size=14, font_family="Lato, sans-serif"))

    if node_type == 'DC':
        fig.update_layout(title = "Delta in units allocated at DCs compared to allocation at actual capacity state:", title_x=0.05)
    elif node_type == 'STR':
        fig.update_layout(title = "Delta in units allocated at Stores compared to allocation at actual capacity state:", title_x=0.05)

    # fig.show()
    st.plotly_chart(fig, use_container_width=True)


def comb_plot_data(data, abs_attr, p_attr):

    abs_attr = label_attr_abs_dict[abs_attr]
    p_attr = label_attr_p_dict[p_attr]

    df_default = data[data['scenario']=="Default"][['crte_d', abs_attr, p_attr]]
    df_default.columns = ['crte_d', 'def_abs', 'def_p']

    df_mechmax=data[data['scenario']=="Mechmax"][['crte_d', abs_attr, p_attr]]
    df_mechmax.columns = ['crte_d', 'mm_abs', 'mm_p']

    df_uncap=data[data['scenario']=="Uncap"][['crte_d', abs_attr, p_attr]]
    df_uncap.columns = ['crte_d', 'unc_abs', 'unc_p']

    plot_df = df_default.merge(df_mechmax, how='left', on=['crte_d']).merge(df_uncap, how='left', on=['crte_d'])
    # plot_df.drop(['scenario_x', 'scenario_y','scenario'], axis = 1, inplace = True)

    return plot_df


def plot_comb_plot(data, abs_attr, p_attr):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    bar_x = comb_plot_data(data, abs_attr, p_attr)['crte_d'].tolist()
    bar_y1 = comb_plot_data(data, abs_attr, p_attr)['def_abs'].tolist()
    bar_y2 = comb_plot_data(data, abs_attr, p_attr)['mm_abs'].tolist()
    bar_y3 = comb_plot_data(data, abs_attr, p_attr)['unc_abs'].tolist()
    line_y1 = comb_plot_data(data, abs_attr, p_attr)['def_p'].tolist()
    line_y2 = comb_plot_data(data, abs_attr, p_attr)['mm_p'].tolist()

    # Add traces
    fig.add_trace(go.Bar(x=bar_x, y=bar_y1, name=abs_attr + ": Default Capacity",marker_color='#b3d1ff'),secondary_y=False,)
    fig.add_trace(go.Bar(x=bar_x, y=bar_y2, name=abs_attr + ": Mechanical Max. Capacity",marker_color='#66a3ff'),secondary_y=False,)
    fig.add_trace(go.Bar(x=bar_x, y=bar_y3, name=abs_attr + ": Uncapacitated Capacity",marker_color='#0052cc'),secondary_y=False,)

    fig.add_trace(go.Scatter(x=bar_x, y=line_y1, name=p_attr + ": Default Capacity",marker_color='#ff8000'),secondary_y=True,)
    fig.add_trace(go.Scatter(x=bar_x, y=line_y2, name=p_attr + ": Mechanical Max. Capacity",marker_color='#ffcc00'),secondary_y=True,)

    # Add figure title
    # fig.update_layout(title_text="<b>SUMMARY METRICS COMPARISON<b>")

    # Set x-axis title
    fig.update_xaxes(title_text="ORDER DATE")

    # Set y-axes titles
    fig.update_yaxes(title_text="TOTAL PACKAGES", secondary_y=False)
    fig.update_yaxes(title_text="PERCENTAGE", secondary_y=True)

    fig.update_layout(autosize=False,width=1000,height=600,)

    plt.tight_layout()
    st.plotly_chart(fig, use_container_width=True)


####################
### INTRODUCTION ###
####################
st.write('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))
with row0_1:
    st.title('NEMO - Order Allocation Simulator')
    # st.caption('A streamlit App by [Jithin Raghavan](https://www.linkedin.com/in/jithinraghavan/)')
with row0_2:
    st.text("")
    # st.caption('Streamlit App by [Jithin Raghavan](https://www.linkedin.com/in/jithinraghavan/)')
    # row0_2.image('Index.jpeg', width = 100)
row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 3.2, .1))
with row3_1:
    st.markdown("NEMO is a simulation tool aiming to become the best-possible user resource for simulating Target's guest order allocation. It is used for studying the impact of input parameters on order allocation and is designed to help business partners in their decision making. The tool has the power to impact inventory placement and positioning, network optimization and labour planning for store operations.")
    st.caption("You can find the source code in the [NEMO Demo Repository](https://github.com/Jithin-Raghavan/NEMO-Demo)")


#################
### SELECTION ###
#################

# st.sidebar.text('')
# st.sidebar.text('')
# st.sidebar.text('')
### SEASON RANGE ###
st.sidebar.markdown("**First select the data range you want to analyze:** 👇")
selected_scenario = st.sidebar.selectbox('Select the scenario for which you want the results displayed', ['Default Capacity','Mechanical Max Capacity','Uncapacitated'])
df_data_filtered_scenario = filter_scenario(dly_loc_df)

# unique_dates = get_unique_dates(dly_loc_df)
# start_date, end_date = st.sidebar.select_slider('Select the date range you want to include', unique_dates, value = ["‏‏‎2022-02-20","2022-02-26"])
start_date = st.sidebar.selectbox('Starting date for simulation', ['2022-05-15'])
end_date = st.sidebar.selectbox('Ending date for simulation', ['2022-05-21'])
# df_data_filtered_date = filter_date(df_data_filtered_scenario)
df_data_filtered_date = df_data_filtered_scenario

### TEAM SELECTION ###
unique_states = get_unique_states(df_data_filtered_date)
all_states_selected = st.sidebar.selectbox('Do you want to only include nodes from specific states? If the answer is yes, please check the box below and then select the state(s) in the new field.', ['Include all available states','Select states manually (choose below)'])
if all_states_selected == 'Select states manually (choose below)':
    selected_states = st.sidebar.multiselect("Select/ Deselect the states you would like to include in the analysis. You can clear the current selection by clicking the corresponding x-button on the right", unique_states, default = unique_states)
df_data_filtered = filter_states(df_data_filtered_date)

### SEE DATA ###
row6_spacer1, row6_1, row6_spacer2 = st.columns((.2, 7.1, .2))
with row6_1:
    st.subheader("Allocation Summary:")

row2_spacer1, row2_1, row2_spacer2, row2_2, row2_spacer3, row2_3, row2_spacer4, row2_4, row2_spacer5   = st.columns((.3, 1.6, 1.3, 1.6, 1.3, 1.6, 1.3, 1.6,0.1))
with row2_1:
    packages_in_df = df_data_filtered['packages'].sum()
    # str_packages = str(packages_in_df) + " Packages"
    str_packages = str(packages_in_df)[0:1] + "," + str(packages_in_df)[1:4] + "," + str(packages_in_df)[4:7] + " Packages"
    st.markdown(str_packages)
with row2_2:
    package_wt_in_df = df_data_filtered['total_lines'].sum()
    # str_package_wt = str(int(round(package_wt_in_df,0))) + " Order-Lines"
    str_package_wt = str(int(round(package_wt_in_df,0)))[0:1] + "," + str(int(round(package_wt_in_df,0)))[1:4] + "," + str(int(round(package_wt_in_df,0)))[4:7] + " Lines"
    st.markdown(str_package_wt)
with row2_3:
    total_units_in_df = df_data_filtered['total_units'].sum()
    # str_units = str(total_units_in_df) + " Units"
    str_units = str(total_units_in_df)[0:1] + "," + str(total_units_in_df)[1:4] + "," + str(total_units_in_df)[4:7] + " Units"
    st.markdown(str_units)
with row2_4:
    total_cost_in_df = df_data_filtered['total_shipcost'].sum()
    # str_cost = "$" + str(int(round(total_cost_in_df,0))) + " Dollars"
    str_cost = "USD " + str(int(round(total_cost_in_df,0)))[0:2] + "," + str(int(round(total_cost_in_df,0)))[2:5] + "," + str(int(round(total_cost_in_df,0)))[5:8]
    st.markdown(str_cost)

row3_spacer1, row3_1, row3_spacer2 = st.columns((.2, 7.1, .2))
with row3_1:
    st.markdown("")
    see_data = st.expander('You can click here to see the raw data first 👉')
    with see_data:
        st.dataframe(data=df_data_filtered.reset_index(drop=True))
st.text('')



# FILTER DATA
summary_df_fin = filter_states(summary_df)
# dly_ov_df_fin = filter_states(dly_ov_df)
# dly_loc_df_fin = filter_states(dly_loc_df)

# LAYING OUT THE MIDDLE SECTION OF THE APP WITH THE MAPS
row4_spacer1, row4_1, row4_spacer1, row4_2 = st.columns((.1,2,.1,2))

with row4_1:
    st.write(f"""**Average throughput in the week with existing capacities**""")
    plot_map_throughput(summary_df_fin,'default_tp_scaled')

with row4_2:
    st.write("**Average throughput in the week with mechanical max capacities**")
    plot_map_throughput(summary_df_fin,'mechmax_tp_scaled')


st.text("")
st.text("")
st.text("")


# ### STATE ###
# row4_spacer1, row4_1, row4_spacer2 = st.columns((.2, 7.1, .2))
# with row4_1:
#     st.subheader('Analysis per State')
# row5_spacer1, row5_1, row5_spacer2, row5_2, row5_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))
# with row5_1:
#     st.markdown('Investigate a variety of stats for nodes from each state. Which state fulfills most of our guest demand? How do stores in each state compare in terms of cost of delivery?')
#     plot_x_per_state_selected = st.selectbox ("Which attribute do you want to analyze?", list(label_attr_dict.keys()), key = 'attribute_state')
#     plot_x_per_state_type = st.selectbox ("Which measure do you want to analyze?", types, key = 'measure_state')
#     specific_state_colors = st.checkbox("Use state specific color scheme")
# with row5_2:
#     if all_states_selected != 'Select states manually (choose below)' or selected_states:
#         plot_x_per_state(plot_x_per_state_selected, plot_x_per_state_type)
#     else:
#         st.warning('Please select at least one state')
#

st.markdown("""---""")

row4_spacer1, row4_1, row4_spacer2 = st.columns((.2, 7.1, .2))
with row4_1:
    st.subheader('Constrained vs. Unconstrained Allocation')

row5_1, row5_spacer2 = st.columns((7.1, .2))
with row5_1:
    # st.markdown("Analysing the delta in units allocated at nodes by node type across different capacity scenarios.")
    plot_delta_units(summary_df_fin,'STR')
    # st.write('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)
    plot_delta_units(summary_df_fin,'DC')

st.markdown("""---""")
#
### SCENARIO ###
row6_spacer1, row6_1, row6_spacer2, row6_2, row6_spacer3, row6_3, row6_spacer4 = st.columns((.2, 2.3, .2, 2.3, .2, 2.3, .2))
with row6_1:
    st.subheader('Summary Metrics')
    st.markdown('Investigate downstream supply chain impact')
with row6_2:

# with row7_1:
    # st.markdown('Investigate downstream impact of capacity scenarios on the supply chain')
    plot_comb_abs = st.selectbox("Which attribute do you want to compare in absolute measures?", list(label_attr_abs_dict.keys()), key = 'attribute_season', index=7)
with row6_3:
    plot_comb_p = st.selectbox("Which attribute do you want to compare in percentage terms?", list(label_attr_p_dict.keys()), key = 'measure_season', index=14)
row7_spacer1, row7_1, row7_spacer2 = st.columns((.1, 7.3, .1))
with row7_1:
    if all_states_selected != 'Select states manually (choose below)' or selected_states:
        plot_comb_plot(dly_ov_df, plot_comb_abs, plot_comb_p)
    else:
        st.warning('Please select at least one state')

st.markdown("""---""")
#
# ### CORRELATION ###
# corr_plot_types = ["Regression Plot (Recommended)","Standard Scatter Plot","Violin Plot (High Computation)"]
#
# row10_spacer1, row10_1, row10_spacer2 = st.columns((.2, 7.1, .2))
# with row10_1:
#     st.subheader('Correlation of Allocation Stats')
# row11_spacer1, row11_1, row11_spacer2, row11_2, row11_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))
# with row11_1:
#     st.markdown('Investigate the correlation of attributes, but keep in mind correlation does not imply causation. Do stores from states that fulfill more packages also incur higher delivery charges?')
#     corr_type = st.selectbox ("What type of correlation plot do you want to see?", corr_plot_types)
#     y_axis_aspect2 = st.selectbox ("Which attribute do you want on the y-axis?", list(label_attr_dict.keys()))
#     x_axis_aspect1 = st.selectbox ("Which attribute do you want on the x-axis?", list(label_attr_dict.keys()),index=8)
# with row11_2:
#     if all_states_selected != 'Select states manually (choose below)' or selected_states:
#         plt_attribute_correlation(x_axis_aspect1, y_axis_aspect2)
#     else:
#         st.warning('Please select at least one state')
#
# for variable in dir():
#     if variable[0:2] != "__":
#         del globals()[variable]
# del variable
