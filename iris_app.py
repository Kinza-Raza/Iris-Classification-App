import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier

st.title("Iris Flower Classification App")
"### This is an app which predicts the **class of iris** based on some parameters. These parameters are shown in the table below. Click on the left to toggle the sidebar, which consists of sliders to modify the values of dimensions of sepal and petal. Have fun playing around!"

st.image("pic.jpg", width=400)
@st.cache
def load_data():
    data = pd.read_csv("iris.data", names=["sepal length cm","sepal width cm","petal length cm","petal width cm","class"])
    return data

data = load_data()

st.sidebar.header("Input Parameters")
st.sidebar.markdown("Drag the slidebars to see the change in predicted output:")

def input_parameters():
    sepal_length = st.sidebar.slider("Sepal Length (cm)", data["sepal length cm"].min(), data["sepal length cm"].max(),
                                     round(data["sepal length cm"].mean(),2))
    sepal_width = st.sidebar.slider("Sepal Width (cm)", data["sepal width cm"].min(), data["sepal width cm"].max(),
                                     round(data["sepal width cm"].mean(),2))
    petal_length = st.sidebar.slider("Petal Length (cm)", data["petal length cm"].min(), data["petal length cm"].max(),
                                     round(data["petal length cm"].mean(),2))
    petal_width = st.sidebar.slider("Petal Width (cm)", data["petal width cm"].min(), data["petal width cm"].max(),
                                     round(data["petal width cm"].mean(),2))
    params = {"sepal length":sepal_length, 
             "sepal width":sepal_width,
             "petal_length":petal_length,
             "petal_width":petal_width}
    features = pd.DataFrame(params, index=[1])
    return features


st.write(" * * * ")
st.markdown("### __User's parameters__:")
parameters = input_parameters()
st.write(parameters)

x = data.iloc[:,0:4].values
y = data.iloc[:,4].values

#Training Decision Tree classifier
model = DecisionTreeClassifier(criterion="entropy",max_depth=6)
model.fit(x,y)
pred = model.predict(parameters)
prob = model.predict_proba(parameters)


progress_bar = st.progress(0)
status_text = st.empty()


for i in range(91):
    progress_bar.progress(i + 10)
    time.sleep(0.005)
    
    if i == 90:
        status_text.text('Successfully processed!')
        "### __Prediction:__"
        st.write(pred[0])

if st.checkbox("Show original dataset"):
    st.write(data)    

"### Want to have a sneak peek of behind-the-scenes? Click on the button below."

if st.button("Details for nerds"):
   
    st.write("We have trained the 'iris' dataset which can be easily downloaded from the UCI Machine Learning repository. The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. Our objective is to predict the class of iris plant on the basis of the length and width of its sepal and petal. The three classes are: ")
    st.markdown("#### 1) Iris Setosa")
    st.markdown("#### 2) Iris Versicolour")
    st.markdown("#### 3) Iris Virginica")
    " "
    st.write("This is a classical example of classification. Although there are multiple classification algorithms available, we have trained our dataset using the Decision Tree Classifier with the parameters: criterion = 'entropy' and max-depth = 6")
 

st.write(" * * * ")
st. markdown("### _References:_")
st.markdown(" [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris) ")