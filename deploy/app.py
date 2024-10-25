# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 23:55:14 2024

@author: Dell
"""

import numpy as np
import pickle 
import streamlit as st

loaded_model = pickle.load(open('C:/Users/Dell/Downloads/sales forecasting 2/deploy/trained_model.sav', 'rb'))

def sales_prediction(input_data):
    
    numpy_array = np.asarray(input_data)
    reshaped = numpy_array.reshape(1,-1)
    pred = loaded_model.predict(reshaped)[0]
    print(pred)
   # print (f"Sales Value is between {pred-714.42} and {pred+714.42}")

def main():
    st.title('Sales Forecasting Engine')
    
    Item_MRP = st.text_input('Enter item MRP:')
    Outlet_Identifier = st.text_input('Enter outlet identifier id:')
    Outlet_Size = st.text_input('Enter the size of the outlet:')
    Outlet_Type = st.text_input('Enter the type of outlet:')
    Outlet_Establishment_Year = st.text_input('Enter the establishment year of the outlet:')
    
    sales = ''
    
    
    if st.button('Sales Test Result'):
        sales = sales_prediction([Item_MRP, Outlet_Identifier, Outlet_Size, Outlet_Type, Outlet_Establishment_Year])
        
    st.success(sales)
    
    
if __name__ == '__main__':
    main()