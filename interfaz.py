
import numpy as np
import pandas as pd
import math

from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import plotly.graph_objects as go
import dash
from dash import dcc
from dash import html
import dash_daq as daq
from dash.dependencies import Input, Output

#X, y = make_regression(n_samples=1000, n_features=6, n_informative=4, random_state=22)
X, y = make_classification(n_samples=1000, n_features=6, n_informative=4, random_state=22)

col_names = ["EDAD","AFECCIÓN PRIN","MUN","Clues","Dias est","Género"]

df = pd.DataFrame(X, columns=col_names)

df["EDAD"]=15*(4+df["EDAD"])
df["Clues"]=10*(4+df["Clues"])
df["Dias est"]=15*(4+df["Dias est"])
df["Género"]=df["Género"]/4
df["MUN"]=2*(4+df["MUN"])
df["AFECCIÓN PRIN"]=2*(4+df["AFECCIÓN PRIN"])
df["Y"] = y

#model = RandomForestRegressor()
model = RandomForestClassifier()
model.fit(df.drop("Y", axis=1), df["Y"])


df_feature_importances = pd.DataFrame(model.feature_importances_*100,columns=["Importance"],index=col_names)
df_feature_importances = df_feature_importances.sort_values("Importance", ascending=False)
# Bar Chart
fig_features_importance = go.Figure()
fig_features_importance.add_trace(go.Bar(x=df_feature_importances.index,
                                         y=df_feature_importances["Importance"],
                                         marker_color='rgb(171, 226, 251)')
                                 )
fig_features_importance.update_layout(title_text='<b>Importancia de los predictores en el modelo<b>', title_x=0.5)


#name, min, mean, max de las 3 important features
slider_1_label = df_feature_importances.index[0]
slider_1_min = math.floor(df[slider_1_label].min())
slider_1_mean = round(df[slider_1_label].mean())
slider_1_max = round(df[slider_1_label].max())
slider_2_label = df_feature_importances.index[1]
slider_2_min = math.floor(df[slider_2_label].min())
slider_2_mean = round(df[slider_2_label].mean())
slider_2_max = round(df[slider_2_label].max())

slider_3_label = df_feature_importances.index[2]
slider_3_min = math.floor(df[slider_3_label].min())
slider_3_mean = round(df[slider_3_label].mean())
slider_3_max = round(df[slider_3_label].max())
app = dash.Dash()


#HTML 
app.layout = html.Div(style={'textAlign': 'center', 'width': '800px', 'font-family': 'Verdana'},
                      
                    children=[

                        
                        html.H1(children="Simulation Tool"),
                        
                        
                        dcc.Graph(figure=fig_features_importance),
                        
                        
                        html.H4(children=slider_1_label),

                        
                        dcc.Slider(
                            id='X1_slider',
                            min=slider_1_min,
                            max=slider_1_max,
                            step=0.5,
                            value=slider_1_mean,
                            marks={i: '{}'.format(i) for i in range(slider_1_min, slider_1_max+1,5)}
                            ),

                    
                        html.H4(children=slider_2_label),

                        dcc.Slider(
                            id='X2_slider',
                            min=slider_2_min,
                            max=slider_2_max,
                            step=0.5,
                            value=slider_2_mean,
                            marks={i: '{}'.format(i) for i in range(slider_2_min, slider_2_max+1,5)}
                        ),

                        html.H4(children=slider_3_label),

                        dcc.Slider(
                            id='X3_slider',
                            min=slider_3_min,
                            max=slider_3_max,
                            step=0.1,
                            value=slider_3_mean,
                            marks={i: '{}'.format(i) for i in np.linspace(slider_3_min, slider_3_max,1+(slider_3_max-slider_3_min)*5)},
                        ),
                        
                        
                        html.H2(id="prediction_result"),

                    ])
                    
@app.callback(Output(component_id="prediction_result",component_property="children"),

              [Input("X1_slider","value"), Input("X2_slider","value"), Input("X3_slider","value")])


def update_prediction(X1, X2, X3):


    input_X = np.array([X1,
                       df["EDAD"].mean(),
                       X2,
                       df["MUN"].mean(),
                       X3,
                       df["AFECCIÓN PRIN"].mean()]).reshape(1,-1)        
    

    prediction = model.predict(input_X)[0]
    

    return "Reingreso: {}".format(round(prediction,1))

if __name__ == "__main__":
    app.run_server()