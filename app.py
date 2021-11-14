import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

teams=['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

cities=['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']


pipe=pickle.load(open('pipe.pkl','rb'))
st.title('Predict The Winning IPL Team')

# st.image('static\images\IPL_background_image.jpg')


st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://wallpapercave.com/wp/wp4059913.jpg");
        background-size: cover;
    }
   .sidebar .sidebar-content {
        background: url("url_goes_here")
        
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1,col2=st.columns(2)

with col1:
    batting_team=st.selectbox('Select the batting Team',sorted(teams))

with col2:
    bowling_team=st.selectbox('Select the bowling Team',sorted(teams))


selected_city=st.selectbox('Select host city',sorted(cities))
target=st.number_input('Target')


col3,col4,col5=st.columns(3)

with col3:
    score=st.number_input('Score')

with col4:
    overs=st.number_input('Overs Completed')

with col5:
    wickets_out=st.number_input('Wicket out')

if st.button('Predict Probability'):
    runs_left=target-score
    balls_left=120-(overs*6)
    wickets_left=10-wickets_out
    crr=score/overs
    rrr=(runs_left*6)/(balls_left)

    input_df=pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],
    'city':[selected_city],'runs_left':[runs_left],'balls_left': [balls_left],
    'wickets_left':[wickets_left],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})


    # st.table(input_df)

    result=pipe.predict_proba(input_df)
    loss=result[0][0]
    win=result[0][1]
    # st.header(batting_team + "-" + str(round(win*100)) +'%')
    # st.header(bowling_team + "-" + str(round(loss*100)) +'%')
    

    labels = [batting_team,bowling_team]
    sizes = [win,loss]
    explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots(figsize=(2,2))
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
    shadow=True, startangle=90)
    ax1.set_title('Winnning Probability of Each Team')
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig1)
