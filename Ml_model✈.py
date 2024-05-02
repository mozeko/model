import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

st.title("Flight Fare Price Prediction")
st.write('This app predicts the **flight fare price!**')

# sidebae
st.sidebar.title("Flight Fare Price Prediction")

st.sidebar.success("Select a page above.")
st.sidebar.markdown("Mad With :heart_eyes: by Analyst.[Zakaria Mostafa](https://www.linkedin.com/in/zakariamostafa/) ")


################################




#from sklearn import datasets




# Loads the Boston House Price Dataset
df_m = pd.read_csv('data_mlnn.csv')
#st.dataframe(df_m.sample(5))

#Loads the Boston House Price Dataset
X = df_m.iloc[:,:-1]
y = df_m.iloc[:,-1]
###
def user_input_features():
    airline = st.sidebar.slider('airline',0,5)
    flight = st.sidebar.slider('flight',0,1500)
    source_city = st.sidebar.slider('source_city',0,5)
    departure_time = st.sidebar.slider('departure_time',0,5)
    stops = st.sidebar.slider('stops', 0,2)
    arrival_time = st.sidebar.slider('arrival_time',0,5)
    destination_city = st.sidebar.slider('destination_city',0,5)
    class_ = st.sidebar.slider('class',0,1)
    duration = st.sidebar.slider('duration', X.duration.min(), X.duration.max(), X.duration.mean())
    days_left = st.sidebar.slider('days_left',0,49)
   
    data = {'airline': airline,
            'flight': flight,
            'source_city': source_city,
            'departure_time': departure_time,
            'stops': stops,
            'arrival_time': arrival_time,
            'destination_city': destination_city,
            'class_': class_,
            'duration': duration,
            'days_left': days_left,
            
            
            }
    features = pd.DataFrame(X, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, y)
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction Of Flight Fare Price')
st.write(prediction)
st.write('---')

#################################
# Splitting the Data into Training set and Testing Set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42)



# Scaling the values to convert the int values to Machine Languages
from sklearn.preprocessing import MinMaxScaler
mmscaler=MinMaxScaler(feature_range=(0,1))

X_train=mmscaler.fit_transform(X_train)
X_test=mmscaler.fit_transform(X_test)
X_train=pd.DataFrame(X_train)
X_test=pd.DataFrame(X_test) 
a={'Model Name':[], 'Mean_Absolute_Error_MAE':[] ,'Adj_R_Square':[] ,'Root_Mean_Squared_Error_RMSE':[] ,'Mean_Absolute_Percentage_Error_MAPE':[] ,'Mean_Squared_Error_MSE':[] ,'Root_Mean_Squared_Log_Error_RMSLE':[] ,'R2_score':[]}
Results=pd.DataFrame(a)
modelrfr = RandomForestRegressor()

#Trainig the model with
modelrfr.fit(X_train, y_train)
    
# Predict the model with test data

y_pred = modelrfr.predict(X_test)
out=pd.DataFrame({'Price_actual':y_test,'Price_pred':y_pred})
result=df_m.merge(out,left_index=True,right_index=True)
#ax = plt.subplots()
fig= plt.figure(figsize=(20,10))
sns.lineplot(data=result,x='days_left',y='Price_actual',color='red')
sns.lineplot(data=result,x='days_left',y='Price_pred',color='blue')
plt.title('Days Left For Departure Versus Actual Ticket Price and Predicted Ticket Price',fontsize=20)
plt.legend(labels=['Price actual','Price predicted'],fontsize=19)
plt.xlabel('Days Left for Departure',fontsize=15)
plt.ylabel('Actual and Predicted Price',fontsize=15)
st.pyplot(fig)