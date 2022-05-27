
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
from matplotlib import pyplot as plt
from  matplotlib.ticker import FuncFormatter
import seaborn as sns
import pydeck as pdk
from pydeck.types import String


st.set_page_config(layout="wide")

### Data Import ###
df_database = pd.read_csv("./data/input_data.csv")
types = ["Mean","Median","Maximum","Minimum","Sum"]

label_attr_dict = {"Packages":"packages","FEDEX Packages":"fedex_packages","UPS Packages":"ups_packages","Other Service Packages":"other_service_packages","No Service Packages":"no_service_packages","Long Zone Packages":"longzone_packages","Zone 2 Packages":"zone_two_packages","Total Zone":"total_zone","Avg. Zone":"average_zone","Total Lines":"total_lines","Average Lines":"average_lines","Total Units":"total_units","Avg. Units":"average_units","Total Shipcost":"total_shipcost","Avg.Shipcost":"average_shipcost","Total Package Weight":"total_package_weight","Avg. Package Weight":"average_package_weight"}

color_dict = {'IL': '#fc4744', 'CA':'#8c0303', 'MO':'#edd134', 'TX':'#fa2323', 'NM':'#cf0c0c', 'OH':'#e62222','VA':'#1f9900', 'MD':'#fff830', 'MI':'#dbca12', 'NJ':'#d10606', 'NY':'#007512', 'TN':'#b50300','IN':'#1c2afc', 'WA':'#eb3838', 'ID':'#061fc2', 'KY':'#127a18', 'PA':'#005ac2', 'DE':'#0707a8','CO':'#d1332e', 'NH':'#0546b5', 'MA':'#265ade', 'RI':'#2b82d9', 'MT':'#f57171','NV':'#38d433','CT':'#10a30b','WV': '#fc4744', 'ME':'#8c0303', 'LA':'#edd134', 'OK':'#fa2323', 'OR':'#cf0c0c','AR':'#e62222', 'KS':'#1f9900', 'MS':'#fff830', 'DC':'#dbca12', 'AK':'#d10606', 'HI':'#007512','VT':'#b50300', 'UT':'#1c2afc', 'FL':'#eb3838', 'AL':'#061fc2', 'MN':'#127a18', 'GA':'#005ac2','SC':'#0707a8', 'AZ':'#d1332e', 'NC':'#0546b5', 'WI':'#265ade', 'IA':'#2b82d9', 'NE':'#f57171','SD':'#38d433', 'ND':'#10a30b','WY': '#fc4744'}

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
    scenario_list = selected_scenario.split()
    df_filtered_scenario = df_data[df_data['scenario'].isin(scenario_list)]

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
def map_ready_data(df_data):
    df = df_data[['lng','lat','units_scaled']]
    map_ready_df = df.reindex(df.index.repeat(df.units_scaled))
    map_ready_df = map_ready_df.drop(columns =['units_scaled']).reset_index(drop=True)
    return map_ready_df

# Plotting the PyDeck Chart
def plot_map_origin(data):
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
                                    map_ready_data(data),
                                    get_position=["lng", "lat"],
                                    auto_highlight=True,
                                    elevation_scale=5000,
                                    pickable=True,
                                    radius = 50000,
                                    elevation_range=[0, 200],
                                    extruded=True,
                                    coverage=1,
                                )
                            ],
                    tooltip={'html': '<b>Number of units (in 1000s):</b> {elevationValue}'}
                ),
            )

def plot_map_fulfill(data):
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
                                    map_ready_data(data),
                                    get_position=["lng", "lat"],
                                    auto_highlight=True,
                                    elevation_scale=5000,
                                    pickable=True,
                                    radius = 50000,
                                    elevation_range=[0, 200],
                                    extruded=True,
                                    coverage=1,
                                )
                            ],
                    tooltip={'html': '<b>Number of units (in 100s):</b> {elevationValue}'}
                ),
            )


####################
### INTRODUCTION ###
####################

row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))
with row0_1:
    st.title('NEMO - Order Allocation Simulator')
    st.caption('A streamlit App by [Jithin Raghavan](https://www.linkedin.com/in/jithinraghavan/)')
with row0_2:
    st.text("")
    # st.caption('Streamlit App by [Jithin Raghavan](https://www.linkedin.com/in/jithinraghavan/)')
    row0_2.image('Index.jpeg', width = 100)
row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 3.2, .1))
with row3_1:
    st.markdown("NEMO is a simulation tool aiming to become the best-possible user resource for simulating Target's guest order allocation. It is used for studying the impact of input parameters on order allocation and is designed to help business partners in their decision making. The tool has the power to impact inventory placement and positioning, network optimization and labour planning for store operations.")
    st.caption("You can find the source code in the [NEMO Demo Repository](https://github.com/Jithin-Raghavan/NEMO-Demo)")


#################
### SELECTION ###
#################

st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
### SEASON RANGE ###
st.sidebar.markdown("**First select the data range you want to analyze:** 👇")
selected_scenario = st.sidebar.selectbox('Select the scenario for which you want the results displayed', ['Baseline','Audit TNT no buffer','MCA TNT no buffer'])
df_data_filtered_scenario = filter_scenario(df_database)

unique_dates = get_unique_dates(df_database)
# start_date, end_date = st.sidebar.select_slider('Select the date range you want to include', unique_dates, value = ["‏‏‎2022-02-20","2022-02-26"])
start_date = st.sidebar.selectbox('Starting date for simulation', ['2022-02-20'])
end_date = st.sidebar.selectbox('Ending date for simulation', ['2022-02-26'])
# df_data_filtered_date = filter_date(df_data_filtered_scenario)
df_data_filtered_date = df_database

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
    str_packages = str(packages_in_df)[0:2] + "," + str(packages_in_df)[2:5] + "," + str(packages_in_df)[5:8] + " Packages"
    st.markdown(str_packages)
with row2_2:
    package_wt_in_df = df_data_filtered['total_package_weight'].sum()
    # str_package_wt = str(int(round(package_wt_in_df,0))) + " gms in weight"
    str_package_wt = str(int(round(package_wt_in_df,0)))[0:2] + "," + str(int(round(package_wt_in_df,0)))[2:5] + "," + str(int(round(package_wt_in_df,0)))[5:8] + " gms in weight"
    st.markdown(str_package_wt)
with row2_3:
    total_units_in_df = df_data_filtered['total_units'].sum()
    # str_units = str(total_units_in_df) + " Units" #68,410,233
    str_units = str(total_units_in_df)[0:2] + "," + str(total_units_in_df)[2:5] + "," + str(total_units_in_df)[5:8] + " Units"
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

#st.dataframe(data=df_stacked.reset_index(drop=True))
#st.dataframe(data=df_data_filtered)

### Data Import ###
origin_demand_df = pd.read_csv("./data/origin_demand.csv")

# LAYING OUT THE MIDDLE SECTION OF THE APP WITH THE MAPS
row4_spacer1, row4_1, row4_spacer1, row4_2 = st.columns((.1,2,.1,2))

with row4_1:
    st.write(f"""**Where is the demand originating from?**""")
    plot_map_origin(origin_demand_df)

with row4_2:
    st.write("**Where is the demand fulfilled from?**")
    plot_map_fulfill(df_data_filtered)

# with row4_1:
#     st.write(f"""**Where is the demand originating from?**""")
#     # layer = pdk.Layer("HeatmapLayer",map_data,opacity=1,get_position=["lng", "lat"],auto_highlight=True,aggregation=String('SUM'),get_weight="total_units")
#     # r = pdk.Deck(layers=[layer])
#     # r.to_html('heatmap.html')
#     row4_1.image('DemandOrigin.png', width = 620)
#
# with row4_2:
#     st.write("**Where is the demand being fulfilled from?**")
#     row4_2.image('DemandFulfilled.png', width = 620)

st.text("")
st.text("")
st.text("")


### STATE ###
row4_spacer1, row4_1, row4_spacer2 = st.columns((.2, 7.1, .2))
with row4_1:
    st.subheader('Analysis per State')
row5_spacer1, row5_1, row5_spacer2, row5_2, row5_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))
with row5_1:
    st.markdown('Investigate a variety of stats for nodes from each state. Which state fulfills most of our guest demand? How do stores in each state compare in terms of cost of delivery?')
    plot_x_per_state_selected = st.selectbox ("Which attribute do you want to analyze?", list(label_attr_dict.keys()), key = 'attribute_state')
    plot_x_per_state_type = st.selectbox ("Which measure do you want to analyze?", types, key = 'measure_state')
    specific_state_colors = st.checkbox("Use state specific color scheme")
with row5_2:
    if all_states_selected != 'Select states manually (choose below)' or selected_states:
        plot_x_per_state(plot_x_per_state_selected, plot_x_per_state_type)
    else:
        st.warning('Please select at least one state')


### SCENARIO ###
row6_spacer1, row6_1, row6_spacer2 = st.columns((.2, 7.1, .2))
with row6_1:
    st.subheader('Analysis per Scenario')
row7_spacer1, row7_1, row7_spacer2, row7_2, row7_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))
with row7_1:
    st.markdown('Investigate developments and trends. Which season had teams score the most goals? Has the amount of passes per games changed?')
    plot_x_per_scenario_selected = st.selectbox ("Which attribute do you want to analyze?", list(label_attr_dict.keys()), key = 'attribute_season')
    plot_x_per_scenario_type = st.selectbox ("Which measure do you want to analyze?", types, key = 'measure_season')
with row7_2:
    if all_states_selected != 'Select states manually (choose below)' or selected_states:
        plot_x_per_scenario(plot_x_per_scenario_selected,plot_x_per_scenario_type)
    else:
        st.warning('Please select at least one state')


### CORRELATION ###
corr_plot_types = ["Regression Plot (Recommended)","Standard Scatter Plot","Violin Plot (High Computation)"]

row10_spacer1, row10_1, row10_spacer2 = st.columns((.2, 7.1, .2))
with row10_1:
    st.subheader('Correlation of Allocation Stats')
row11_spacer1, row11_1, row11_spacer2, row11_2, row11_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))
with row11_1:
    st.markdown('Investigate the correlation of attributes, but keep in mind correlation does not imply causation. Do stores from states that fulfill more packages also incur higher delivery charges?')
    corr_type = st.selectbox ("What type of correlation plot do you want to see?", corr_plot_types)
    y_axis_aspect2 = st.selectbox ("Which attribute do you want on the y-axis?", list(label_attr_dict.keys()))
    x_axis_aspect1 = st.selectbox ("Which attribute do you want on the x-axis?", list(label_attr_dict.keys()),index=8)
with row11_2:
    if all_states_selected != 'Select states manually (choose below)' or selected_states:
        plt_attribute_correlation(x_axis_aspect1, y_axis_aspect2)
    else:
        st.warning('Please select at least one state')

for variable in dir():
    if variable[0:2] != "__":
        del globals()[variable]
del variable
