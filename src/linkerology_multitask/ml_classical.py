from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
import plotly.graph_objects as go
from typing import Tuple, List
from linkerology_multitask import LABEL_MAPPER


POSSIBLE_TYPES = ['xgboost', 'rf']

def train_binary_model_with_grid_search(X_train:np.ndarray, Y_train:np.ndarray, X_val:np.ndarray, Y_val:np.ndarray,
                                        targets:List[str], model_type:str, param_grid:dict) -> Tuple[List[float], List[float]]:
    if model_type not in POSSIBLE_TYPES:
        raise ValueError(f'Argument type must be one of {POSSIBLE_TYPES}. Received "{model_type}"')
    
    accuracy_scores = []
    f1_scores = []
    for target_i in range(Y_train.shape[1]):
        print(f'Target: {LABEL_MAPPER[targets[target_i]]}')

        train_nan_mask = ~np.isnan(Y_train[:, target_i])

        if np.all(Y_train[:, target_i][train_nan_mask] == 1) or np.all(Y_train[:, target_i][train_nan_mask] == 0): # \
            # only one class so can't build model
            accuracy_scores.append(np.nan)
            f1_scores.append(np.nan)
        else:
            if model_type == 'xgboost':
                model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
            else: # random forest
                model = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(model, param_grid, cv=3, verbose=1)
            grid_search.fit(X_train[train_nan_mask],
                            Y_train[:, target_i][train_nan_mask])
            
            print(f'Best Hyperparameters: {grid_search.best_params_}')
            best_model = grid_search.best_estimator_
            best_model.fit(X_train[train_nan_mask],
                           Y_train[:, target_i][train_nan_mask])
            
            val_nan_mask = ~np.isnan(Y_val[:, target_i])

            y_pred = best_model.predict(X_val[val_nan_mask])

            accuracy_scores.append(accuracy_score(Y_val[:, target_i][val_nan_mask], y_pred))
            f1_scores.append(f1_score(Y_val[:, target_i][val_nan_mask], y_pred))
    return accuracy_scores, f1_scores

def plot_results(accuracy_scores:List[float], f1_scores:List[float], targets:List[str], title:str) -> go.Figure:
    targets = [LABEL_MAPPER[t] for t in targets]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=targets,
        y=accuracy_scores,
        name='Accuracy',
        text=[round(x, 2) for x in accuracy_scores],
        textposition='auto'
    ))
    fig.add_trace(go.Bar(
        x=targets,
        y=f1_scores,
        name='F1',
        text=[round(x, 2) for x in f1_scores],
        textposition='auto'
    ))
    fig.update_layout(width=800, height=600, title=title, title_x=0.5)
    return fig
