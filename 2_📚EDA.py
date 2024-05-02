#loading libraries
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


st.sidebar.success("Select a page above.")
st.sidebar.write("")
st.sidebar.write("Filter Your Data:")
cat_filt = st.sidebar.selectbox("Categorical",[None,'airline','stops','class'])

# loadind data
df = pd.read_csv('new_data.csv')



#body
df1=df.groupby(['flight','airline'],as_index=False).count()

# row a
a1, a2, a3, a4, a5 = st.columns(5)
a1.metric("Count. Indigo",df[df['airline']=='Indigo']['airline'].count())
a2.metric("Count.GO_FIRST",df[df['airline']=='GO_FIRST']['airline'].count())
a3.metric("Count.AirAsia",df[df['airline']=='AirAsia']['airline'].count())
a4.metric("Count.SpiceJet",df[df['airline']=='SpiceJet']['airline'].count())
a5.write("The most popular companies")
st.write("")
# row b
#fig = px.line(data_frame=df,x='days_left',y='price',color='class')
#st.plotly_chart(fig, use_container_width=True)
st.write("Ticket Price Versus Flight Duration Based on Class")
fig =plt.figure(figsize=(20,10))
sns.lineplot(data=df,x="duration",y="price",hue="class",palette="hls")
st.pyplot(fig)

fig = px.box(data_frame=df,x= 'airline',y='price',title='Airlines Vs Price',color=cat_filt)
st.plotly_chart(fig,use_container_width=True)

fig = px.box(data_frame=df,x='stops',y='price',title='Stops Vs Ticket Price',color=cat_filt)
st.plotly_chart(fig,use_container_width=True)



# rowc 
c1 ,c2   =st.columns((5,5))

with c1:
    st.text("Flights Count of Different Airlines")
    fig = px.bar(data_frame=df,x='airline',y='flight',color=cat_filt)
    st.plotly_chart(fig,use_container_width=True)


with c2 :
    st.text("Classes of Different Airlines")
    fig = px.pie(data_frame=df,names='class',values='price',color=cat_filt)
    st.plotly_chart(fig,use_container_width=True)


but = st.button("Show Data")
if but:
    st.dataframe(df.sample(5))