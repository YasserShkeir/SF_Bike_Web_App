import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import matplotlib.pyplot as plt
from requests import options
import streamlit as st
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split as tts

st.set_page_config(page_title='Data Scientist Assessment', layout='wide')

if 'bool_t1' not in st.session_state:
    st.session_state['bool_t1']=False

if 'bool_t2' not in st.session_state:
    st.session_state['bool_t2']=False

if 'bool_t3' not in st.session_state:
    st.session_state['bool_t3']=False

st.write('Please choose a Task:')

col1, col2, col3 = st.columns(3)

with col1:
    if st.button(' Task 1 '):
        st.session_state['bool_t1']=True
        st.session_state['bool_t2']=False
        st.session_state['bool_t3']=False

with col2:
    if st.button(' Task 2 '):
        st.session_state['bool_t1']=False
        st.session_state['bool_t2']=True
        st.session_state['bool_t3']=False

with col3:
    if st.button(' Task 3 '):
        st.session_state['bool_t1']=False
        st.session_state['bool_t2']=False
        st.session_state['bool_t3']=True     

def Task1():
    data_money = pd.read_csv('task1_data_Edited_Money.csv')
    data_money["Start_Date"] = pd.to_datetime(data_money["Start_Date"], format="%d/%m/%Y").dt.date
    data_money.sort_values("Start_Date", inplace=True)

    if 'bool_t1_v1' not in st.session_state:
        st.session_state['bool_t1_v1']=False

    if 'bool_t1_v2' not in st.session_state:
        st.session_state['bool_t1_v2']=False

    if 'bool_t1_v3' not in st.session_state:
        st.session_state['bool_t1_v3']=False

    with col1:
        # Headers
        st.write("""
        # Task 1
        # """)
        view = st.selectbox('\nPlease choose a view:', ('','Money View', 'Station Insights','Fleet Trends'))
        if view == 'Money View':
            st.session_state['bool_t1_v1']=True
            st.session_state['bool_t1_v2']=False
            st.session_state['bool_t1_v3']=False
        if view == 'Station Insights':
            st.session_state['bool_t1_v1']=False
            st.session_state['bool_t1_v2']=True
            st.session_state['bool_t1_v3']=False
        if view == 'Fleet Trends':
            st.session_state['bool_t1_v1']=False
            st.session_state['bool_t1_v2']=False
            st.session_state['bool_t1_v3']=True

    def money_view():
        st.header("Money Section:")
        st.write("Sample of the Data:")
        st.write(data_money.iloc[0:1001])

        data_money_cust = data_money.loc[(data_money['user_type'] == 'Customer')]
        data_money_subs = data_money.loc[(data_money['user_type'] == 'Subscriber')]

        ### START Pricing Data For Customers
        def calc_cust(x):
            return 2 + (-(-(x-30)/15)) * 3

        data_money_cust['trip_price'] = np.where(data_money_cust["duration_min"]<=30, 2, data_money_cust["duration_min"].apply(calc_cust))
        ### END Pricing Data For Customers

        ### START Pricing Data For Subs
        result = data_money_subs.groupby('bike_id').agg({'Start_Date': ['min', 'max']})

        result['days_subscribed'] = result.Start_Date['max'] - result.Start_Date['min']
        result['days_subscribed'] = 1 + result['days_subscribed'].dt.days

        def ceil_fun(x):
            return -(-x/30)

        result['monthly_fees'] =  result['days_subscribed'].apply(ceil_fun) * 15

        result = result.drop(['Start_Date', 'days_subscribed'], axis=1)

        result['price_per_trip'] = result['monthly_fees'] / (data_money_subs['bike_id'].value_counts().sort_index())

        def round_fun(x):
            return round(x, 2)

        result['price_per_trip'] = result['price_per_trip'].apply(round_fun)

        data_money_subs = data_money_subs.merge(result,how='left', left_on='bike_id', right_on='bike_id')
        
        def calc_subs():
            for i in range(len(data_money_subs)):
                if data_money_subs['duration_min'].iloc[i] > 45:
                    data_money_subs[('price_per_trip','')].iloc[i] = data_money_subs[('price_per_trip','')].iloc[i] + (-(-(data_money_subs['duration_min'].iloc[i]-45)/15)) * 3
            
        calc_subs()

        subs_chart_data = data_money_subs.groupby(["Start_Date"]).sum().reset_index()
        cust_chart_date = data_money_cust.groupby(["Start_Date"]).sum().reset_index()

        fig, ax = plt.subplots(1, figsize=(12,5))

        plt.plot(cust_chart_date['Start_Date'], cust_chart_date['trip_price'], label="Customers Daily Profit",markersize=1, color='r', linewidth=1)
        plt.plot(subs_chart_data['Start_Date'], subs_chart_data[('price_per_trip', '')], label="Subscribers Daily Profit",markersize=1, color='b', linewidth=1)
        plt.plot(subs_chart_data['Start_Date'], cust_chart_date['trip_price']+subs_chart_data[('price_per_trip', '')], label="All Users Daily Profit",markersize=1, color='g', linewidth=1)
        plt.legend(loc="upper right")

        plt.show()

        st.write('Here we can see the daily, monthly, and yearly profit trends of the business')
        st.plotly_chart(fig)

    def station_insights():
        data_stations = pd.DataFrame().assign(
                    date=data_money['Start_Date'],
                    start_station_name=data_money['start_station_name'],
                    start_longitude=data_money['start_station_longitude'],
                    start_latitude=data_money['start_station_latitude'],
                    dest_station_name=data_money['end_station_name'],
                    dest_longitude=data_money['end_station_longitude'],
                    dest_latitude=data_money['end_station_latitude'],
                    ez_count=1)

        st.header('Station Insights:')
        col1, col2 = st.columns(2)

        with col1:
            chosen_station = st.selectbox('Please select a station for insights: ', data_stations['start_station_name'].unique())
            data_stations_strt = data_stations.loc[data_stations['start_station_name'] == chosen_station]
            data_stations_dest = data_stations.loc[data_stations['dest_station_name'] == chosen_station]

            # st.success('The overall number of trips starting at station ' + chosen_station + ' is ' + str(len(data_stations_strt)))
            # st.error('The overall number of trips ending at station ' + chosen_station + ' is ' + str(len(data_stations_dest)))

            chart_data_strt = data_stations_strt.groupby(['date']).sum().reset_index()
            chart_data_strt = chart_data_strt.drop(['start_longitude', 'start_latitude', 'dest_longitude', 'dest_latitude'], axis = 1)

            chart_data_dest = data_stations_dest.groupby(['date']).sum().reset_index()
            chart_data_dest = chart_data_dest.drop(['start_longitude', 'start_latitude', 'dest_longitude', 'dest_latitude'], axis = 1)

            fig, ax = plt.subplots(1)
            plt.plot(chart_data_strt['date'], chart_data_strt['ez_count'], label=chosen_station + " as a starting station",markersize=1, color='g', linewidth=1)
            plt.plot(chart_data_dest['date'], chart_data_dest['ez_count'], label=chosen_station + " as an ending station",markersize=1, color='r', linewidth=1)
            plt.legend(loc="upper right")
            plt.title('Daily Incoming and Outcoming trips to station ' + chosen_station)

            plt.show()

            st.plotly_chart(fig)
            

        with col2:
            st.write('Sample Map of destination stations when starting at station ' + chosen_station)
            data_stations_strt = data_stations_strt.drop(['ez_count', 'date'], axis = 1)
            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/light-v9',
                initial_view_state=pdk.ViewState(
                    latitude=37.76,
                    longitude=-122.4,
                    zoom=11,
                    pitch=50,
                ),
                layers=[
                    pdk.Layer(
                        'HexagonLayer',
                        data=data_stations_strt,
                        get_position='[dest_longitude, dest_latitude]',
                        radius=200,
                        elevation_scale=4,
                        elevation_range=[0, 1000],
                        pickable=True,
                        extruded=True,
                    ),
                    pdk.Layer(
                        'ScatterplotLayer',
                        data=data_stations_strt,
                        get_position='[start_longitude, start_latitude]',
                        get_color='[30, 200, 100, 160]',
                        get_radius=300,
                    ),
                ],
            ))
            st.caption('Starting station marked as green, destination stations are the hexagons')
   
    def fleet_trends():
        st.header('Fleet Trends:')

        data_fleet = pd.DataFrame().assign(
            duration_min=data_money['duration_min'],
            bike_id=data_money['bike_id'],
            trip_count=1
        )
        
        data_fleet = data_fleet.groupby(['bike_id']).sum().reset_index()
        
        col1, col2 = st.columns(2)

        data_money['week_day'] = pd.to_datetime(data_money['Start_Date'])
        data_money['week_day'] = data_money['week_day'].dt.day_name()

        data_fleet_week_day = pd.DataFrame().assign(
            week_day=data_money['week_day'],
            duration_min=data_money['duration_min'],
            trip_count=1
        )

        data_fleet_week_day['week_day']=pd.Categorical(
            data_fleet_week_day['week_day'],
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

        data_fleet_week_day = data_fleet_week_day.groupby(['week_day']).sum().reset_index()

        fig1 = px.bar(data_fleet_week_day, x='week_day', y='duration_min', title='Overall rental time for each week day')
        fig2 = px.bar(data_fleet_week_day, x='week_day', y='trip_count', title='Overall trip count for each week day')
        

        with col1:
            data_fleet_count = data_fleet.groupby(['bike_id']).sum().reset_index().sort_values(by=['trip_count'], ascending=False)
            st.write('The top 100 bikes by trip count: ', data_fleet_count.iloc[0:99])
            st.plotly_chart(fig1)

        with col2:
            data_fleet_mins = data_fleet.groupby(['bike_id']).sum().reset_index().sort_values(by=['duration_min'], ascending=False)
            st.write('The top 100 bikes by sum of trips duration (in minutes): ', data_fleet_mins.iloc[0:99])
            st.plotly_chart(fig2)   

    if st.session_state['bool_t1_v1']:
        money_view()
    if st.session_state['bool_t1_v2']:
        station_insights()
    if st.session_state['bool_t1_v3']:
        fleet_trends()

def Task2():
    # Headers
    st.write("""
    # Task 2
    # """)
    st.header("Predicting a trip duration based on user input, using RandomForestRegressor:")
    df = pd.read_csv('https://drive.google.com/file/d/1cX1VCcUH7TODu90fxZpRwAOWbc3SBrO4/view?usp=sharing')
    df["start_time"] = pd.to_datetime(df["start_time"])
    
    st.write("Sample of the data:", df.iloc[0:1000])
    st.markdown('Dataset shape: ')
    st.info(df.shape)

    ##### Making df smaller for testing faster T.T
    col1, col2, col3 = st.columns(3)
    with col2:
        st.write('Choose the length of data for your model')
        length_data_model = st.slider('Smaller Values: Faster & Less Accurate', 100000, 2350000, 100000, 5000)
        df = df.iloc[0:length_data_model]

    ### Editing Values
    df.dropna(inplace=True)

    cleanup_vals = {"user_type":     {"Customer": 0, "Subscriber": 1},
                    "member_gender": {"Male": 0, "Female": 1, "Other": 2, "No Answer": 3}}
    
    df = df.replace(cleanup_vals)

    df = df.drop(['start_station_name'], axis=1)

    for column in df:
        df[column] = pd.to_numeric(df[column], downcast='float')
    ###

    ### Editing eval data
    eval_df = pd.read_csv('https://drive.google.com/file/d/1p0IoumrWmjGf_gY4YkvD-comdZPI6qxz/view?usp=sharing')
    eval_df=eval_df.dropna()
    eval_df["start_time"] = pd.to_datetime(eval_df["start_time"])
    eval_df = eval_df.replace(cleanup_vals)
    eval_df = eval_df.drop(['start_station_name'], axis=1)
    for column in eval_df:
        eval_df[column] = pd.to_numeric(eval_df[column], downcast='float')
    ###

    y=df['duration_sec']
    x=df.drop('duration_sec',axis=1)
    
    x_train,x_test,y_train,y_test=tts(x,y,test_size=len(eval_df)/length_data_model)

    # st.write(
    #     'eval_df: '+ str(len(eval_df)),eval_df,
    #     'x_train: '+ str(len(x_train)),x_train,
    #     'x_test: '+ str(len(x_test)), x_test,
    #     'y_train: '+ str(len(y_train)), y_train,
    #     'y_test: '+ str(len(y_test)), y_test)

    rf = RandomForestRegressor(n_estimators=15, random_state=0)
    rf.fit(x_train, y_train)  

    y_pred = rf.predict(x_test)

    st.write('Accuracy Metrics using splitted data:')
    task2_col1, task2_col2, task2_col3 = st.columns(3)
    with task2_col1:
        st.write('Mean Absolute Error:')
        st.info(metrics.mean_absolute_error(y_test, y_pred))
    with task2_col2:
        st.write('Mean Squared Error:')
        st.info(metrics.mean_squared_error(y_test, y_pred))
    with task2_col3:
        st.write('Root Mean Squared Error:')
        st.info(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    
    # prediction input
    with st.form(key='input_form'):
        st.write('Input your data:')
        test2_col1, test2_col2, test2_col3 = st.columns(3)
        with test2_col1:
            test_time=st.time_input('Choose starting time:')
            test_stid=st.selectbox('Choose station ID:', np.unique(df['start_station_id']).astype('int32'))
            test_bday=st.text_input('Member Birth Year:')
            
        with test2_col2:
            test_date=st.date_input('Choose starting date:')
            test_bkid=st.selectbox('Choose bike ID:', np.unique(df['bike_id']).astype('int32'))
            test_gndr=st.selectbox('Member Gender:', ('Male', 'Female'))

        with test2_col3:
            test_user=st.radio('User Type:', ('Customer', 'Subscriber'))

        predict_data={
            'start_time':[test_time],
            'start_station_id':[test_stid],
            'bike_id':[test_bkid],
            'user_type':[test_user],
            'member_birth_year':[test_bday],
            'member_gender':[test_gndr],
            'start_date':[test_date]
        }

        predict_data= pd.DataFrame(predict_data)

        ### Match input date and time to the models datetime 
        predict_data['start_date'] = pd.to_datetime(predict_data['start_date'].astype(str))
        predict_data['start_time'] = pd.to_datetime(predict_data['start_time'].astype(str))

        def to_date(x):
            x = str(x)
            return x[0:11]

        predict_data['start_date'] = predict_data['start_date'].apply(to_date)

        def to_time(x):
            x = str(x)
            return x[11:]

        predict_data['start_time'] = predict_data['start_time'].apply(to_time)

        dt = pd.to_datetime(predict_data['start_date'] + predict_data['start_time'])
        predict_data.drop(['start_date','start_time'], axis=1, inplace=True)
        predict_data.insert(0, 'start_time', dt)

        ###

        ### Get Longitude and Latitude from station ID
        df = df.drop(['duration_sec', 'start_time', 'bike_id', 'user_type', 'member_birth_year', 'member_gender'], axis = 1)
        new_df = df.groupby(by='start_station_id').mean()

        test_latitude = new_df['start_station_latitude'].loc[test_stid]
        test_longitude = new_df['start_station_longitude'].loc[test_stid]

        predict_data.insert(2, 'start_station_latitude', test_latitude)
        predict_data.insert(3, 'start_station_longitude', test_longitude)


        if st.form_submit_button('Confirm'):
            predict_data = predict_data.replace(cleanup_vals)
            for column in predict_data:
                predict_data[column] = pd.to_numeric(predict_data[column], downcast='float')
            msg = 'The predicted duration is ' + str(rf.predict(predict_data)[0].astype('int')) + ' seconds.'
            st.error(msg)
            
def Task3():
    # Headers
    st.write("""
    # Task 3
    # """)
    st.header("Predicting bike demand for the next week, using RandomForestRegressor:")

if st.session_state['bool_t1']:
    Task1()

if st.session_state['bool_t2']:
    Task2()

if st.session_state['bool_t3']:
    Task3()
