import streamlit as st
import pickle
import numpy as np
import math
pipe=pickle.load(open('pipe.pkl','rb'))
data=pickle.load(open('data.pkl','rb'))
st.title('Laptop Predictor')

# brand
company=st.selectbox('Brand',data['Company'].unique())
 
 #type of laptop
type=st.selectbox('type',data['TypeName'].unique())

 # ram
ram=st.selectbox('RAM(in GB)',[2,4,6,8,12,16,32,64])

# GPU
gpu=st.selectbox('GPU',data['Gpu'].unique())
#Weight
weight=st.number_input('weight of th laptop')

# Touchscreen
touchscreen=st.selectbox('Touchscreen',['No','Yes'])

# ips
ips=st.selectbox('ips',['No','Yes'])

#screen size
screen_size=st.number_input('screen size')
#resolution
resolution=st.selectbox('screen Resolution',['1920x1080','1366x768','1600x900','3840x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu=st.selectbox('CPU',data['Cpu brand'].unique())

hdd=st.selectbox('HDD(in GB)',[0,128,256,512,1024,2028])
ssd=st.selectbox('SSD(in GB)',[0,128,256,512,1024,2028])

os=st.selectbox('Os',data['os'].unique())

if st.button('Predict Price'):
    #query
    
    if touchscreen =='Yes':
        touchscreen=1
    else:
        touchscreen=0

    if ips=='Yes':
        ips=1

    else:
        ips=0
    
    x_res=int(resolution.split('x')[0])
    y_res=int(resolution.split('x')[1])
    ppi=((x_res**2)+(y_res**2))**0.5/screen_size
    query=np.array([company,type,ram,gpu,weight,touchscreen,ips,ppi,cpu,hdd,ssd,os])
    query=np.array(query,dtype=object)
    query=query.reshape(1,12)
    st.title('The predicted price of this configuration is:'+str(int(np.exp(pipe.predict(query)[0]))))
