# !pip install -r requirements.txt
# https://kamranshaikh1993-ml-deploy-index-1-1bkuz8.streamlit.app/

import streamlit as st

import pickle

file1 = open("model.pkl",'rb') 
file2 = open("CountVector.pkl",'rb')
model = pickle.load(file1)  
cv = pickle.load(file2)  

# msg = 'It was amazing. I liked it a lot'

def predict_review(msg):   
    msg_cv = cv.transform([msg])
    result = model.predict(msg_cv)[0]
    return result


st.write("# Hello World ")
st.write("This is :blue[test]")

# st.write("# Hello World ")

message = st.text_area("Message")

if st.button("Predict"):
    result = predict_review(message)
    st.write(result)
    
    
    
    # Font
#--------------------------

# new_title = '<p style="font-family:sans-serif; color:Green; font-size: 72px;">New image</p>'
# st.markdown(new_title, unsafe_allow_html=True)
# st.result(message, channels="BGR")

#----------------------------
