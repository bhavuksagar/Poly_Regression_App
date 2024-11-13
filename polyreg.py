import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import pandas as pd


st.title("Welcome to Ploynomial Regression App!")
st.subheader("Visual represntation of effect of degree")

st.write(
         '''
         Polynomial Regression is a regression algorithm that models the relationship between a dependent(y) and independent variable(x) as nth degree polynomial. The Polynomial Regression equation is given below:
''')
st.write("y= b0+b1x1+ b2x12+ b2x13+...... bnx1n")
st.write('''
It is also called the special case of Multiple Linear Regression in ML. Because we add some polynomial terms to the Multiple Linear regression equation to convert it into Polynomial Regression.
It is a linear model with some modification in order to increase the accuracy.
The dataset used in Polynomial regression for training is of non-linear nature.
It makes use of a linear regression model to fit the complicated and non-linear functions and datasets.
         
         
         '''
         )


st.subheader("Graph")
degree=int(st.slider("Enter the Degree Value",min_value=1,max_value=10,value=2))
st.write(degree)

x=6* np.random.randn(500,1) - 3
y=0.5+x**2+1.5 * x +  np.random.randn(500,1)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=123)



def pipline_poly(deg):
    poly=PolynomialFeatures(degree=deg,include_bias=True)
    model1=LinearRegression()
    model1.fit(x_train,y_train)
    model2=LinearRegression()
    y_pre_non=model1.predict(x_test)

    pipe=Pipeline([("poly",poly),("model",model2)])

    pipe.fit(x_train,y_train)
    y_pre_poly=pipe.predict(x_test)

    return pipe,y_pre_non,y_pre_poly

pipe,y_pre_non,y_pre_poly=pipline_poly(degree)


df=pd.DataFrame({
    "Feature":x_train.reshape(400),
    "Target":y_train.reshape(400),
    "Poly_Pre":pipe.predict(x_train).reshape(400)
})


score=r2_score(y_test,pipe.predict(x_test))
st.scatter_chart(data=df,x="Feature",y="Poly_Pre",y_label="Predicted")

st.subheader("R2 Score")
st.write(f"Score is:{score}")






    





