
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



df = pd.read_csv("entrena.csv",sep=';')
df.sample(10)

def cal_prevalance(y_actual):
    return (sum(y_actual)/len(y_actual))

print(f"Prevalance : {round(cal_prevalance(df.REINGRESO.values)*100, 3)} %")

df.columns

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df["CLUES_str"] = encoder.fit_transform(df["CLUES_str"])
df["AFECPRIN_str"] = encoder.fit_transform(df["AFECPRIN_str"])

df_data = df

df_data = df_data.sample(n = len(df_data), random_state = 42)
df_data = df_data.reset_index(drop = True)

df_valid_test = df_data.sample(frac = 0.3, random_state = 42)
df_test = df_valid_test.sample(frac = 0.3, random_state = 42)
df_valid = df_valid_test.drop(df_test.index)
df_train_all = df_data.drop(df_valid_test.index)

# split the training data into positive and negative
rows_pos = df_train_all.REINGRESO == 1
df_train_pos = df_train_all.loc[rows_pos]
df_train_neg = df_train_all.loc[~rows_pos]

# merge the balanced data
df_train = pd.concat([df_train_pos, df_train_neg.sample(n = len(df_train_pos), random_state = 42)],axis = 0)

# shuffle the order of training samples 
df_train = df_train.sample(n = len(df_train), random_state = 42).reset_index(drop = True)

cols2use = ['CLUES_str', 'DIASESTA_int', 'EDAD_int', 'SEXO_int', 'AFECPRIN_str',
       'MUNIC_int']
       
X_train = df_train[cols2use].values
X_train_all = df_train_all[cols2use].values
X_valid = df_valid[cols2use].values

y_train = df_train['REINGRESO'].values
y_valid = df_valid['REINGRESO'].values

from sklearn.preprocessing import StandardScaler
import pickle

ss = StandardScaler()
ss.fit(X_train_all)

scalerfile = 'StndSclr.sav'
pickle.dump(ss, open(scalerfile, 'wb'))

# load it back
ss = pickle.load(open(scalerfile, 'rb'))

X_train_tf = ss.transform(X_train)
X_valid_tf = ss.transform(X_valid)

# Creating helper functions
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

def calc_specificity(y_actual, y_pred, thresh):
    # calculates specificity
    return sum((y_pred < thresh) & (y_actual == 0)) /sum(y_actual ==0)

def print_report(y_actual, y_pred, thresh):
    
    auc = roc_auc_score(y_actual, y_pred)
    accuracy = accuracy_score(y_actual, (y_pred > thresh))
    recall = recall_score(y_actual, (y_pred > thresh))
    precision = precision_score(y_actual, (y_pred > thresh))
    specificity = calc_specificity(y_actual, y_pred, thresh)
    print('AUC:%.3f'%auc)
    print('accuracy:%.3f'%accuracy)
    print('recall:%.3f'%recall)
    print('precision:%.3f'%precision)
    print('specificity:%.3f'%specificity)
    print('prevalence:%.3f'%cal_prevalance(y_actual))
    print(' ')
    return auc, accuracy, recall, precision, specificity

thresh = 0.5

    # Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth = 10, random_state = 42)
tree.fit(X_train_tf, y_train)

y_train_preds = tree.predict_proba(X_train_tf)[:,1]
y_valid_preds = tree.predict_proba(X_valid_tf)[:,1]

print('Decision Tree')
print('Training:')
tree_train_auc, tree_train_accuracy, tree_train_recall, tree_train_precision, tree_train_specificity =print_report(y_train,y_train_preds, thresh)
print('Validation:')
tree_valid_auc, tree_valid_accuracy, tree_valid_recall, tree_valid_precision, tree_valid_specificity = print_report(y_valid,y_valid_preds, thresh)

#Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(max_depth = 6, random_state = 42)
rf.fit(X_train_tf, y_train)

y_train_preds = rf.predict_proba(X_train_tf)[:,1]
y_valid_preds = rf.predict_proba(X_valid_tf)[:,1]

print('Random Forest')
print('Training:')
rf_train_auc, rf_train_accuracy, rf_train_recall, rf_train_precision, rf_train_specificity =print_report(y_train,y_train_preds, thresh)
print('Validation:')
rf_valid_auc, rf_valid_accuracy, rf_valid_recall, rf_valid_precision, rf_valid_specificity = print_report(y_valid,y_valid_preds, thresh)

df_feature_importances = pd.DataFrame(rf.feature_importances_*100,columns=["Importance"],index=cols2use)
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

slider_4_label = df_feature_importances.index[3]
slider_4_min = math.floor(df[slider_4_label].min())
slider_4_mean = round(df[slider_4_label].mean())
slider_4_max = round(df[slider_4_label].max())

app = dash.Dash()


#HTML 
app.layout = html.Div(style={'textAlign': 'center', 'width': '800px', 'font-family': 'Verdana'},
                      
                    children=[

                        
                        html.H1(children="PREDICCIÓN DE REINGRESOS"),
                        
                        
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
                        
                        html.H4(children=slider_4_label),

                        dcc.Slider(
                            id='X4_slider',
                            min=slider_4_min,
                            max=slider_4_max,
                            step=0.1,
                            value=slider_4_mean,
                            marks={i: '{}'.format(i) for i in np.linspace(slider_4_min, slider_4_max,1+(slider_4_max-slider_4_min)*5)},
                        ),
                        
                        html.H2(id="prediction_result"),

                    ])
                    
@app.callback(Output(component_id="prediction_result",component_property="children"),

              [Input("X1_slider","value"), Input("X2_slider","value"), Input("X3_slider","value"), Input("X4_slider","value")])


def update_prediction(X1, X2, X3,X4):


    input_X = np.array([X1, df["EDAD_int"].mean(), X2,X3, X4, df["EDAD_int"].mean()]).reshape(1,-1)        
    

    prediction = tree.predict(input_X)[0]
    

    return "Reingreso: {}".format(round(prediction,1))

if __name__ == "__main__":
    app.run_server()