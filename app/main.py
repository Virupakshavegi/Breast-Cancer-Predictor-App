import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np



def preprocessed_data():
    data=pd.read_csv("data/data.csv")
    # print(data.head())
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
  
    data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
    return data
def add_sidebar():
  st.sidebar.header("Cell Nuclei Measurements")
  data=preprocessed_data()
  labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]
  inputdict={}
  for label, key in labels:
      inputdict[key]=st.sidebar.slider(
          label,
          min_value=float(0),
          max_value=float(data[key].max()),
          value=float(data[key].mean())
    )

  return inputdict


def scaledvalues(inputdict):
    data=preprocessed_data()
    X=data.drop(['diagnosis'],axis=1)
    scaleddict={}

    for key,value in inputdict.items():
        maxval=X[key].max()
        minval=X[key].min()
        scaledvalue=(value-minval)/(maxval-minval)
        scaleddict[key]=scaledvalue
    return scaleddict

def radarchart(inputdata):

    inputdata=scaledvalues(inputdata)

    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
          inputdata['radius_mean'], inputdata['texture_mean'], inputdata['perimeter_mean'],
          inputdata['area_mean'], inputdata['smoothness_mean'], inputdata['compactness_mean'],
          inputdata['concavity_mean'], inputdata['concave points_mean'], inputdata['symmetry_mean'],
          inputdata['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
          inputdata['radius_se'], inputdata['texture_se'], inputdata['perimeter_se'], inputdata['area_se'],
          inputdata['smoothness_se'], inputdata['compactness_se'], inputdata['concavity_se'],
          inputdata['concave points_se'], inputdata['symmetry_se'],inputdata['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
          inputdata['radius_worst'], inputdata['texture_worst'], inputdata['perimeter_worst'],
          inputdata['area_worst'], inputdata['smoothness_worst'], inputdata['compactness_worst'],
          inputdata['concavity_worst'], inputdata['concave points_worst'], inputdata['symmetry_worst'],
          inputdata['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
  ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
            showlegend=True
    )
    return fig
    
def prediction(inputdata):
    # model = pickle.load(open('model/model.pkl', 'rb'))
    # scaler = pickle.load(open('model/scaler.pkl', 'rb'))
    with open('model/model.pkl', 'rb') as f:
        model=pickle.load(f)
    with open('model/scaler.pkl', 'rb') as f:
        scaler=pickle.load(f)
    
    input_array = np.array(list(inputdata.values())).reshape(1, -1)
    input_arrayscaled=scaler.transform(input_array)
    pred=model.predict(input_arrayscaled)
    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is:")
    # st.write(pred)

    if pred[0]==0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)


    st.write("Probability of being benign: ", model.predict_proba(input_arrayscaled)[0][0])
    st.write("Probability of being malicious: ", model.predict_proba(input_arrayscaled)[0][1])
    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")
    
    


    

def main():
    st.set_page_config(
        page_title="Breast Cancer Prediction",
        page_icon=":female-doctor",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    inputdata=add_sidebar()
    # st.write(inputdata)
 
    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar. ")
  
    col1, col2 = st.columns([4,1])
    with col1:
        # st.write("this is column 1")
        chart=radarchart(inputdata)
        st.plotly_chart(chart)
    
    with col2:
        prediction(inputdata)
        # st.write("this is column 2")



if __name__ == "__main__":
    main()