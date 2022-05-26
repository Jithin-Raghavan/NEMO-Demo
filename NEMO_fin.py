###########################################################################################################
###########################################################################################################
###########################################################################################################
######## author = Jithin Raghavan
######## website = https://www.linkedin.com/in/jithinraghavan/
######## layout inspired by https://share.streamlit.io/tylerjrichards/streamlit_goodreads_app/books.py
###########################################################################################################
###########################################################################################################
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

color_dict = {'1.0': '#fc4744', '2.0':'#8c0303', '3':'#edd134', '4':'#fa2323', '5':'#cf0c0c', '6':'#e62222','7':'#1f9900', '8':'#fff830', '9':'#dbca12', '10':'#d10606', '11':'#007512', '12':'#b50300','13':'#1c2afc', '14':'#eb3838', '15':'#061fc2', '16':'#127a18', '17':'#005ac2', '18':'#0707a8','19':'#d1332e', '20':'#0546b5', '21':'#265ade', '22':'#2b82d9', '23':'#f57171','24':'#38d433','25':'#10a30b','26': '#fc4744', '27':'#8c0303', '28':'#edd134', '29':'#fa2323', '30':'#cf0c0c','31':'#e62222', '32':'#1f9900', '33':'#fff830', '34':'#dbca12', '35':'#d10606', '36':'#007512','37':'#b50300', '38':'#1c2afc', '39':'#eb3838', '40':'#061fc2', '41':'#127a18', '42':'#005ac2','43':'#0707a8', '44':'#d1332e', '45':'#0546b5', '46':'#265ade', '47':'#2b82d9', '48':'#f57171','49':'#38d433', '50':'#10a30b','51': '#fc4744','NA':'#edd134'}

### Helper Methods ###
def get_unique_scenarios(df_data):
    #Returns unique date range for labels
    return np.unique(df_data.scenario).tolist()

def get_unique_dates(df_data):
    #returns minimum and maximum
    return np.unique(df_data.order_date).tolist()

def get_unique_states(df_data):
    states = list(np.unique(np.array(df_data.state)))
    states = [x for x in states if str(x) != 'nan']
    unique_states = [int(x) for x in states]
    return unique_states

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
          'axes.facecolor':'#0e1117',
          'axes.edgecolor': '#0e1117',
          'axes.labelcolor': 'white',
          'figure.facecolor': '#0e1117',
          'patch.edgecolor': '#0e1117',
          'text.color': 'white',
          'xtick.color': 'white',
          'ytick.color': 'white',
          'grid.color': 'grey',
          'font.size' : 8,
          'axes.labelsize': 12,
          'xtick.labelsize': 8,
          'ytick.labelsize': 12}
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

    ax.set(xlabel = "Scenario", ylabel = y_str)
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
          'axes.facecolor':'#0e1117',
          'axes.edgecolor': '#0e1117',
          'axes.labelcolor': 'white',
          'figure.facecolor': '#0e1117',
          'patch.edgecolor': '#0e1117',
          'text.color': 'white',
          'xtick.color': 'white',
          'ytick.color': 'white',
          'grid.color': 'grey',
          'font.size' : 8,
          'axes.labelsize': 12,
          'xtick.labelsize': 8,
          'ytick.labelsize': 12}

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
          'axes.facecolor':'#0e1117',
          'axes.edgecolor': '#0e1117',
          'axes.labelcolor': 'white',
          'figure.facecolor': '#0e1117',
          'patch.edgecolor': '#0e1117',
          'text.color': 'white',
          'xtick.color': 'white',
          'ytick.color': 'white',
          'grid.color': 'grey',
          'font.size' : 8,
          'axes.labelsize': 12,
          'xtick.labelsize': 8,
          'ytick.labelsize': 12}
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



####################
### INTRODUCTION ###
####################

# t1, t2 = st.columns((0.07,0.05))
# t1.image('Index.jpeg', width = 100)

row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))
with row0_1:
    st.title('NEMO - Order Allocation Simulator')
with row0_2:
    st.text("")
    st.subheader('Streamlit App by [Jithin Raghavan](https://www.linkedin.com/in/jithinraghavan/)')
row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 3.2, .1))
with row3_1:
    st.markdown("NEMO is a simulation tool aiming to become the best-possible user resource for simulating Target's guest order allocation. It is used for studying the impact of input parameters on order allocation and is designed to help business partners in their decision making. The tool has the power to impact inventory placement and positioning, network optimization and labour planning for store operations.")
    st.markdown("You can find the source code in the [NEMO ExploreAI Repository](https://git.target.com/JithinRaghavan/NEMO-AI-Conference)")


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

row2_spacer1, row2_1, row2_spacer2, row2_2, row2_spacer3, row2_3, row2_spacer4, row2_4, row2_spacer5   = st.columns((.2, 1.6, .2, 1.6, .2, 1.6, .2, 1.6, .2))
with row2_1:
    packages_in_df = df_data_filtered['packages'].sum()
    str_packages = "🏟️ " + str(packages_in_df) + " Packages"
    st.markdown(str_packages)
with row2_2:
    package_wt_in_df = df_data_filtered['total_package_weight'].sum()
    str_package_wt = str(round(package_wt_in_df,2)) + " Grams"
    st.markdown(str_package_wt)
with row2_3:
    total_units_in_df = df_data_filtered['total_units'].sum()
    str_units = "🥅 " + str(total_units_in_df) + " Units"
    st.markdown(str_units)
with row2_4:
    total_cost_in_df = df_data_filtered['total_shipcost'].sum()
    str_cost = str(round(total_cost_in_df,2)) + " Dollars"
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


# LAYING OUT THE MIDDLE SECTION OF THE APP WITH THE MAPS
row4_spacer1, row4_1, row4_spacer1, row4_2 = st.columns((.1,2,.1,2))
map_data = df_data_filtered[['lng','lat','total_units']]

# SETTING THE ZOOM LOCATIONS FOR THE AIRPORTS
# chicago = [39.8097343, -98.5556199]
# zoom_level = 3
# midpoint = mpoint(map_data["lat"], map_data["lng"])

with row4_1:
    st.write(f"""**Where is the demand originating from?**""")
    # layer = pdk.Layer("HeatmapLayer",map_data,opacity=1,get_position=["lng", "lat"],auto_highlight=True,aggregation=String('SUM'),get_weight="total_units")
    # r = pdk.Deck(layers=[layer])
    # r.to_html('heatmap.html')
    row4_1.image('DemandOrigin.png', width = 650)

with row4_2:
    st.write("**Where is our demand being fulfilled from?**")
    row4_2.image('DemandFulfilled.png', width = 650)



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

# ### SCENARIO ###
# row6_spacer1, row6_1, row6_spacer2 = st.columns((.2, 7.1, .2))
# with row6_1:
#     st.subheader('Analysis per Season')
# row7_spacer1, row7_1, row7_spacer2, row7_2, row7_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))
# with row7_1:
#     st.markdown('Investigate developments and trends. Which season had teams score the most goals? Has the amount of passes per games changed?')
#     plot_x_per_scenario_selected = st.selectbox ("Which attribute do you want to analyze?", list(label_attr_dict.keys()), key = 'attribute_season')
#     plot_x_per_scenario_type = st.selectbox ("Which measure do you want to analyze?", types, key = 'measure_season')
# with row7_2:
#     if all_states_selected != 'Select states manually (choose below)' or selected_states:
#         plot_x_per_scenario(plot_x_per_scenario_selected,plot_x_per_scenario_type)
#     else:
#         st.warning('Please select at least one state')
#
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
#     x_axis_aspect1 = st.selectbox ("Which attribute do you want on the x-axis?", list(label_attr_dict.keys()))
# with row11_2:
#     if all_states_selected != 'Select states manually (choose below)' or selected_states:
#         plt_attribute_correlation(x_axis_aspect1, y_axis_aspect2)
#     else:
#         st.warning('Please select at least one state')

### SCENARIO ###
row6_spacer1, row6_1, row6_spacer2 = st.columns((.2, 7.1, .2))
with row6_1:
    st.subheader('Analysis per Season')
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
