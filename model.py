import pandas as pd
import numpy as np
import streamlit as st
import pickle
import xgboost as xgb
import plotly.express as px 
from PIL import Image 


st.set_page_config(page_title="Car Dealer Prediction", page_icon="🚗")
    
tab1, tab2, tab3, tab4 = st.tabs(["BMW Data Exploration", "Mercedes Data Exploration",
                                  "BMW Price Prediction", 'Mercedes Price Prediction'])


with tab1:
    st.subheader('Data Exploration for BMW Cars')
    bmw_df = pd.read_excel('BMW for Streamlit filtering.xlsx')
    st.dataframe(bmw_df.head(5), hide_index=True)

    st.markdown("---")
    st.subheader("Filter the Data")

    start_price, end_price = st.select_slider(
        "Select a range of Prices",
        options=sorted(bmw_df['Qiymet'].unique()),
        value=(bmw_df['Qiymet'].min(), bmw_df['Qiymet'].max())
    )

    options_city = st.multiselect(
    "Choose cities",
    options=sorted(bmw_df['Sheher'].unique()),
    default=None)

    options_model = st.multiselect(
    "Choose car models",
    options=sorted(bmw_df['Model'].unique()),
    default=None)

    new_options = ['Hamısı'] + list(bmw_df['Yeni?'].unique())
    option_new = st.selectbox(
    "New or Old?",
    options=new_options
)

    options_year = None
    if option_new != 'Bəli':
        options_year = st.multiselect(
        "Choose years:",
        options=sorted(bmw_df['Buraxilish ili'].unique(), reverse=True),
        default=None)
        
    kredit_options = st.radio('Kredit?', options = ['hamısı'] + list(bmw_df['Kredit'].unique()), horizontal=True)
    barter_options = st.radio('Barter?', options = ['hamısı'] + list(bmw_df['Barter'].unique()), horizontal=True)


    filtered_df = bmw_df[(bmw_df['Qiymet'] >= start_price) & (bmw_df['Qiymet'] <= end_price)]

    if options_city:
        filtered_df = filtered_df[filtered_df['Sheher'].isin(options_city)]

    if options_model:
        filtered_df = filtered_df[filtered_df['Model'].isin(options_model)]

    if options_year:
        filtered_df = filtered_df[filtered_df['Buraxilish ili'].isin(options_year)]

    if option_new != 'Hamısı':
        filtered_df = filtered_df[filtered_df['Yeni?'] == option_new]
        
    if kredit_options == 'Var':
        filtered_df = filtered_df[filtered_df['Kredit'] == 'var']
    elif kredit_options == 'Yox':
        filtered_df = filtered_df[filtered_df['Kredit'] == 'yox']

    if barter_options == 'Var':
        filtered_df = filtered_df[filtered_df['Barter'] == 'var']

    elif barter_options == 'Yox':
        filtered_df = filtered_df[filtered_df['Barter'] == 'yox']

    st.markdown("---")
    st.subheader(f"Filtered Results: {len(filtered_df)} cars found")
    st.dataframe(filtered_df, hide_index=True)    
    
    
    for _, row in filtered_df.iterrows():
        with st.container():
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(row["Shekil urli"], width='stretch')
            
            with col2:
                st.markdown(f"""
                    <div style="border:1px solid #ddd; border-radius:10px; padding:10px; background:white; margin-bottom:15px;">
                        <h3 style="margin:0; color:#111;">{row['Qiymet']:,} AZN</h3>
                        <p style="margin:0; font-weight:bold;">BMW {row['Model']}</p>
                        <p style="margin:0;">{row['Buraxilish ili']}, {row['Motor gucu']} L, {row['Km']} km</p>
                        <p style="margin:0; color:gray; font-size:14px;">{row['Sheher']}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.link_button("For more information on site, click that:", row["Url"])
                
                
with tab2:
    st.subheader('Data Exploration for Mercedes-Benz Cars')
    try:
        mercedes_df = pd.read_excel('Mercedes for Streamlit filtering.xlsx')
    except FileNotFoundError:
        st.error("Error: 'Mercedes for Streamlit.xlsx' not found. Please create this file with your Mercedes data.")
        st.stop()

    st.dataframe(mercedes_df.head(5), hide_index=True)
    
    st.markdown("---")
    st.subheader('Filter the data')
    
    
    start_price_2, end_price_2 = st.select_slider(
        'Select a price range',
        options=sorted(mercedes_df['Qiymet'].unique()),
        value=(mercedes_df['Qiymet'].min(), mercedes_df['Qiymet'].max()),
        key='price_slider_mb'
    )

    options_city_2 = st.multiselect(
        'Choose cities',
        options = sorted(mercedes_df['Sheher'].unique()),
        default=None,
        key='city_multiselect_mb'
        )
    
    options_model_2 = st.multiselect(
        'Choose car model(s)',
        options = sorted(mercedes_df['Model'].unique()),
        default = None,
        key='model_multiselect_mb'
    )
    
    new_options_2 = ['Hamısı'] + list(mercedes_df['Yeni?'].unique())
    
    options_new_2 = st.selectbox(
        'New or Old?',
        options=new_options_2,
        key='new_old_selectbox_mb' 
    )
    
    options_year_2 = None
    if options_new_2 != 'Bəli':
        options_year_2 = st.multiselect(
            'Choose year(s)',
            options = sorted(mercedes_df['Buraxilish ili'].unique(), reverse=True),
            default = None,
            key='year_multiselect_mb'
        )
    
    kredit_options_2 = st.radio('Kredit?', options = ['hamısı'] + list(mercedes_df['Kredit'].unique()), horizontal=True, key='kredit_radio_mb')
    barter_options_2 = st.radio('Barter?', options = ['hamısı'] + list(mercedes_df['Barter'].unique()), horizontal=True, key='barter_radio_mb')

        
    filtered_df_mb = mercedes_df[(mercedes_df['Qiymet'] >= start_price_2) & (mercedes_df['Qiymet'] <= end_price_2)]
    
    if options_city_2:
        filtered_df_mb = filtered_df_mb[filtered_df_mb['Sheher'].isin(options_city_2)]
        
    if options_model_2:
        filtered_df_mb = filtered_df_mb[filtered_df_mb['Model'].isin(options_model_2)]
        
    if options_year_2:
        filtered_df_mb = filtered_df_mb[filtered_df_mb['Buraxilish ili'].isin(options_year_2)]
        
    if options_new_2 != 'Hamısı':
        filtered_df_mb = filtered_df_mb[filtered_df_mb['Yeni?'] == options_new_2]
        
    if kredit_options_2 == 'Var':
        filtered_df_mb = filtered_df_mb[filtered_df_mb['Kredit'] == 'var']
    elif kredit_options_2 == 'Yox':
        filtered_df_mb = filtered_df_mb[filtered_df_mb['Kredit'] == 'yox']
        
    if barter_options_2 == 'Var':
        filtered_df_mb = filtered_df_mb[filtered_df_mb['Barter'] == 'var']
    elif barter_options_2 == 'Yox':
        filtered_df_mb = filtered_df_mb[filtered_df_mb['Barter'] == 'yox']
        
    st.markdown("---")
    st.subheader(f"Filtered Results: {len(filtered_df_mb)} cars found")
    st.dataframe(filtered_df_mb, hide_index=True)
        
    for _, row in filtered_df_mb.iterrows():
        with st.container():
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(row["Shekil urli"], width='stretch')
            
            with col2:
                st.markdown(f"""
                    <div style="border:1px solid #ddd; border-radius:10px; padding:10px; background:white; margin-bottom:15px;">
                        <h3 style="margin:0; color:#111;">{row['Qiymet']:,} AZN</h3>
                        <p style="margin:0; font-weight:bold;">Mercedes {row['Model']}</p>
                        <p style="margin:0;">{row['Buraxilish ili']}, {row['Motor gucu']} L, {row['Km']} km</p>
                        <p style="margin:0; color:gray; font-size:14px;">{row['Sheher']}</p>
                    </div>
                """, unsafe_allow_html=True)

                st.link_button("For more information on site, click that:", row["Url"])


with tab3:
    st.subheader('BMW Price Prediction')
    st.markdown("In this section you can predict the price of a BMW car based on its features using a best-trained model.")

    @st.cache_resource
    def load_bmw_pipeline(series_name):
        filename = f'pipeline_gb_{series_name.replace(" ", "_")}.sav'
        try:
            with open(filename, 'rb') as file:
                pipeline = pickle.load(file)
            return pipeline
        except FileNotFoundError:
            return None

    st.markdown("### Input Car Features")

    bmw_df_model = pd.read_excel('BMW for Streamlit modelling.xlsx')

    series_options = [s for s in bmw_df_model['Series'].unique().astype(str) if s != 'nan']
    selected_series = st.selectbox("Select Car Series", sorted(series_options), key='bmw_series_pred')

    bmw_pipeline = load_bmw_pipeline(selected_series)

    if bmw_pipeline is None:
        st.error(f"A trained model pipeline for '{selected_series}' was not found. Please train and save it first.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            bmw_df_model['Series'] = bmw_df_model['Series'].astype(str)
            model = st.selectbox("Model", sorted(bmw_df_model[bmw_df_model['Series']==selected_series]['Model'].unique()), key='bmw_model_pred')
            engine_size = st.number_input("Engine Size (L)", min_value=1.0, max_value=6.0, step=1.0, value=2.0, key='bmw_engine_pred')
            km = st.number_input("Km", min_value=0, max_value=650000, step=1000, value=0, key='bmw_km_pred')
            year = st.number_input("Year of Manufacture", min_value=1984, max_value=2025, step=1, value=2015, key='bmw_year_pred')
            horsepower = st.number_input("Horsepower", min_value=50, max_value=600, step=10, value=150, key='bmw_hp_pred')
            seat = st.selectbox("Number of Seats", sorted(bmw_df_model['Yerlerin sayi'].unique()), key='bmw_seat_pred')
        with col2:
            ban_type = st.selectbox("Body Type", sorted(bmw_df_model['Ban novu'].unique()), key='bmw_ban_pred')
            gearbox = st.selectbox("Gear box", sorted(bmw_df_model['Suretler qutusu'].unique()), key='bmw_gearbox_pred')
            gear = st.selectbox("Gear", sorted(bmw_df_model['Oturucu'].unique()), key='bmw_gear_pred')
            gearbox_type = st.selectbox("Gearbox Type", sorted(bmw_df_model['Yanacaq novu'].unique()), key='bmw_gearboxtype_pred')

        if st.button("Predict Price", key='bmw_predict_button'):
            input_data = pd.DataFrame({
                'Model': [model],
                'Motor gucu': [engine_size],
                'Km': [km],
                'Buraxilish ili': [year],
                'Yerlerin sayi': [seat],
                'At gucu': [horsepower],
                'Ban novu': [ban_type],
                'Suretler qutusu': [gearbox],
                'Oturucu': [gear],
                'Yanacaq novu': [gearbox_type]
            })

            predicted_price = bmw_pipeline.predict(input_data)[0]

            st.success(f"Predicted Price: {predicted_price:,.0f} AZN")

        
with tab4:
    st.subheader('Mercedes Benz Price Prediction')
    st.markdown("In this section you can predict the price of a Mercedes car based on its features using a best-trained model.")

    @st.cache_resource
    def load_mercedes_pipeline(series_name):
        filename = f'pipeline_xgb_{series_name.replace(" ", "_")}.sav'
        try:
            with open(filename, 'rb') as file:
                pipeline = pickle.load(file)
            return pipeline
        except FileNotFoundError:
            return None

    st.markdown("### Input Car Features")

    mercedes_df_model = pd.read_excel('Mercedes for Streamlit modelling.xlsx')

    series_options = [s for s in mercedes_df_model['Series'].unique().astype(str) if s != 'nan']
    selected_series = st.selectbox("Select Car Series", sorted(series_options), key='merc_series_pred')

    mercedes_pipeline = load_mercedes_pipeline(selected_series)

    if mercedes_pipeline is None:
        st.error(f"A trained model pipeline for '{selected_series}' was not found. Please train and save it first.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            mercedes_df_model['Series'] = mercedes_df_model['Series'].astype(str)
            model = st.selectbox("Model", sorted(mercedes_df_model[mercedes_df_model['Series']==selected_series]['Model'].unique()), key='merc_model_pred')
            engine_size = st.number_input("Engine Size (L)", min_value=0.0, max_value=6.3, step=1.0, value=2.0, key='merc_engine_pred')
            km = st.number_input("Km", min_value=0, max_value=1000000, step=1000, value=0, key='merc_km_pred')
            year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, step=1, value=2015, key='merc_year_pred')
            horsepower = st.number_input("Horsepower", min_value=75, max_value=900, step=10, value=150, key='merc_hp_pred')
            seat = st.selectbox("Number of Seats", sorted(mercedes_df_model['Yerlerin sayi'].unique()), key='merc_seat_pred')
        with col2:
            ban_type = st.selectbox("Body Type", sorted(mercedes_df_model['Ban novu'].unique()), key='merc_ban_pred')
            gearbox = st.selectbox("Gear box", sorted(mercedes_df_model['Suretler qutusu'].unique()), key='merc_gearbox_pred')
            gear = st.selectbox("Gear", sorted(mercedes_df_model['Oturucu'].unique()), key='merc_gear_pred')
            gearbox_type = st.selectbox("Gearbox Type", sorted(mercedes_df_model['Yanacaq novu'].unique()), key='merc_gearboxtype_pred')

        if st.button("Predict Price", key='merc_predict_button'):
            input_data = pd.DataFrame({
                'Model': [model],
                'Motor gucu': [engine_size],
                'Km': [km],
                'Buraxilish ili': [year],
                'Yerlerin sayi': [seat],
                'At gucu': [horsepower],
                'Ban novu': [ban_type],
                'Suretler qutusu': [gearbox],
                'Oturucu': [gear],
                'Yanacaq novu': [gearbox_type]
            })

            predicted_price = mercedes_pipeline.predict(input_data)[0]

            st.success(f"Predicted Price: {predicted_price:,.0f} AZN")



                    
                    
                    
                    
                    