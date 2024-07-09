import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import dash_daq as daq

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Model Evaluator"), className="mb-2")
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Label("Choose a model"),
            dcc.Dropdown(id='model_option', options=[
                {"label": "Custom", "value": "Custom"},
                {"label": "VGG16", "value": "VGG16"},
                {"label": "VGG19", "value": "VGG19"},
                {"label": "ResNet", "value": "ResNet"},
                {"label": "SimpleCNN", "value": "SimpleCNN"}
            ], value='Custom')
        ], width=6),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Label("Upload your network model (.onnx or .pt/.pth)"),
            dcc.Upload(id='upload_network', children=html.Div(['Drag and Drop or ', html.A('Select Files')]), multiple=False)
        ], width=6, id='network_upload_div'),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Label("Upload your input image"),
            dcc.Upload(id='upload_image', children=html.Div(['Drag and Drop or ', html.A('Select Files')]), multiple=False)
        ], width=6),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Label("Number of workers"),
            dcc.Input(id='num_worker', type='number', value=1, min=0)
        ], width=3),
        dbc.Col([
            dbc.Label("Backend"),
            dcc.Dropdown(id='back_end', options=[
                {"label": "cpu", "value": "cpu"},
                {"label": "cuda", "value": "cuda"}
            ], value='cpu')
        ], width=3),
        dbc.Col([
            dbc.Label("Number of symbols"),
            dcc.Input(id='num_symbol', type='text', value='Full')
        ], width=3),
        dbc.Col([
            dbc.Label("Noise level"),
            dcc.Input(id='noise', type='number', value=0.00001)
        ], width=3),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Label("Available RAM per worker"),
            dcc.Input(id='RAM', type='number', value=1.0)
        ], width=3),
        dbc.Col([
            dbc.Label("Resize input image"),
            daq.BooleanSwitch(id='resize_input', on=True)
        ], width=3),
        dbc.Col([
            dbc.Label("Resize width"),
            dcc.Input(id='resize_width', type='number', value=224, min=1)
        ], width=3, id='resize_width_div'),
        dbc.Col([
            dbc.Label("Resize height"),
            dcc.Input(id='resize_height', type='number', value=224, min=1)
        ], width=3, id='resize_height_div'),
    ]),
    dbc.Row([
        dbc.Col(dbc.Button("Prepare Evaluation", id='prepare_evaluation', color='primary'), width=3),
    ]),
    dbc.Row([
        dbc.Col(html.Div(id='evaluation_status'), width=12),
    ]),
    dbc.Row([
        dbc.Col(dbc.Button("Execute Evaluation", id='execute_evaluation', color='success'), width=3),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='evaluation_results'), width=12),
    ]),
    dbc.Row([
        dbc.Col(dbc.Button("Previous", id='prev_button', color='secondary'), width=2),
        dbc.Col(dbc.Button("Next", id='next_button', color='secondary'), width=2),
    ]),
    dbc.Row([
        dbc.Col(html.Div(id='relevance_image'), width=12),
    ]),
])

@app.callback(
    Output('network_upload_div', 'style'),
    Input('model_option', 'value')
)
def toggle_network_upload(model_option):
    if model_option == 'Custom':
        return {'display': 'block'}
    else:
        return {'display': 'none'}

@app.callback(
    Output('resize_width_div', 'style'),
    Output('resize_height_div', 'style'),
    Input('resize_input', 'on')
)
def toggle_resize_input(resize_input):
    if resize_input:
        return {'display': 'block'}, {'display': 'block'}
    else:
        return {'display': 'none'}, {'display': 'none'}

@app.callback(
    Output('evaluation_status', 'children'),
    Output('evaluation_results', 'figure'),
    Output('relevance_image', 'children'),
    Input('prepare_evaluation', 'n_clicks'),
    Input('execute_evaluation', 'n_clicks'),
    State('model_option', 'value'),
    State('upload_network', 'contents'),
    State('upload_image', 'contents'),
    State('num_worker', 'value'),
    State('back_end', 'value'),
    State('num_symbol', 'value'),
    State('noise', 'value'),
    State('RAM', 'value'),
    State('resize_input', 'on'),
    State('resize_width', 'value'),
    State('resize_height', 'value'),
    prevent_initial_call=True
)
def evaluate_model(prepare_clicks, execute_clicks, model_option, upload_network, upload_image, num_worker, back_end, num_symbol, noise, RAM, resize_input, resize_width, resize_height):
    ctx = dash.callback_context

    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'prepare_evaluation':
        if upload_image is not None:
            content_type, content_string = upload_image.split(',')
            input_image = base64.b64decode(content_string)
            files = {
                'input_image': BytesIO(input_image)
            }
            if model_option == "Custom" and upload_network is not None:
                content_type, content_string = upload_network.split(',')
                network_model = base64.b64decode(content_string)
                files['network'] = BytesIO(network_model)
                model_name = "custom"
            else:
                model_name = model_option.lower()

            try:
                num_symbol_value = int(num_symbol)
            except ValueError:
                num_symbol_value = str(num_symbol)

            data = {
                'model_name': model_name,
                'num_worker': num_worker,
                'back_end': back_end,
                'num_symbol': num_symbol_value,
                'noise': noise,
                'RAM': RAM,
                'resize_input': resize_input,
                'resize_width': resize_width if resize_input else None,
                'resize_height': resize_height if resize_input else None
            }

            response = requests.post("http://localhost:8000/prepare_evaluation/", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                messages = result["messages"]
                status = [html.Div(html.P(message)) for message in messages]

                if resize_input:
                    image = Image.open(BytesIO(input_image))
                    resized_image = image.resize((resize_width, resize_height))
                    buffer = BytesIO()
                    resized_image.save(buffer, format="PNG")
                    resized_image_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    status.append(html.Img(src='data:image/png;base64,{}'.format(resized_image_str)))

                return status, dash.no_update, dash.no_update

            else:
                return [html.Div(html.P(f"Error: {response.status_code} - {response.text}"))], dash.no_update, dash.no_update
        else:
            return [html.Div(html.P("Please upload the input image."))], dash.no_update, dash.no_update

    elif button_id == 'execute_evaluation':
        response = requests.post("http://localhost:8000/execute_evaluation/")
        if response.status_code == 200:
            result = response.json()

            argmax = result["argmax"]
            true_values = [float(x) for x in result["true"]]
            center_values = [float(x) for x in result["center"]]
            min_values = [float(x) for x in result["min"]]
            max_values = [float(x) for x in result["max"]]

            x = np.arange(len(argmax))  # Les indices des classes

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.bar(x - 0.2, true_values, 0.4, label='True', color='green')
            ax.bar(x, center_values, 0.4, label='Center', color='yellow')
            ax.errorbar(x, center_values, yerr=[np.array(center_values) - np.array(min_values), np.array(max_values) - np.array(center_values)], fmt='o', color='red', label='Min/Max')

            ax.set_xlabel('Class')
            ax.set_ylabel('Values')
            ax.set_title('Evaluation Results')
            ax.set_xticks(x)
            ax.set_xticklabels(argmax)
            ax.legend()

            evaluation_fig = fig

            relevance_images = result['relevance']
            num_images = len(relevance_images)
            relevance_index = 0
            relevance_image_array = np.array(relevance_images[relevance_index]) / noise
            relevance_image_array = (relevance_image_array * 255).astype(np.uint8)
            relevance_image = Image.fromarray(relevance_image_array).resize((resize_width, resize_height))

            original_image = Image.open(BytesIO(input_image)).resize((resize_width, resize_height)).convert("RGBA")
            relevance_image = relevance_image.convert("RGBA")
            
            blended_image = Image.blend(original_image, relevance_image, alpha=0.5)
            buffer = BytesIO()
            blended_image.save(buffer, format="PNG")
            blended_image_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

            relevance_image_div = html.Div([
                html.Img(src='data:image/png;base64,{}'.format(blended_image_str)),
                html.P(f'Relevance Image {relevance_index + 1}/{num_images}')
            ])

            return dash.no_update, evaluation_fig, relevance_image_div

        else:
            return [html.Div(html.P(f"Error: {response.status_code} - {response.text}"))], dash.no_update, dash.no_update


@app.callback(
    Output('relevance_image', 'children'),
    Input('prev_button', 'n_clicks'),
    Input('next_button', 'n_clicks'),
    State('relevance_image', 'children'),
    State('upload_image', 'contents'),
    State('resize_width', 'value'),
    State('resize_height', 'value'),
    State('noise', 'value'),
    prevent_initial_call=True
)
def update_relevance_image(prev_clicks, next_clicks, relevance_image_div, upload_image, resize_width, resize_height, noise):
    if relevance_image_div:
        relevance_index = int(relevance_image_div[1].children.split()[2].split('/')[0]) - 1
        num_images = int(relevance_image_div[1].children.split()[2].split('/')[1])
        
        if 'prev_button' in dash.callback_context.triggered[0]['prop_id']:
            relevance_index = (relevance_index - 1) % num_images
        else:
            relevance_index = (relevance_index + 1) % num_images

        relevance_images = [relevance_image_div[0].src]  # This should be replaced by actual relevance images list from response
        relevance_image_array = np.array(relevance_images[relevance_index]) / noise
        relevance_image_array = (relevance_image_array * 255).astype(np.uint8)
        relevance_image = Image.fromarray(relevance_image_array).resize((resize_width, resize_height))

        content_type, content_string = upload_image.split(',')
        input_image = base64.b64decode(content_string)
        original_image = Image.open(BytesIO(input_image)).resize((resize_width, resize_height)).convert("RGBA")
        relevance_image = relevance_image.convert("RGBA")
        
        blended_image = Image.blend(original_image, relevance_image, alpha=0.5)
        buffer = BytesIO()
        blended_image.save(buffer, format="PNG")
        blended_image_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return html.Div([
            html.Img(src='data:image/png;base64,{}'.format(blended_image_str)),
            html.P(f'Relevance Image {relevance_index + 1}/{num_images}')
        ])
    else:
        raise dash.exceptions.PreventUpdate


if __name__ == '__main__':
    app.run_server(debug=True)
