import base64
import io
import numpy as np  # Added numpy for matrix operations
from dash import Dash, dcc, html, Input, Output, State, callback, no_update
from PIL import Image, ImageOps, ImageFilter
import cv2 
from model import *
# =========================================
# 1. MODEL LOADING & LOGIC
# =========================================

# TODO: Load your real models here (Global Scope) so they only load once when the app starts.
# Example:
# model_side = torch.load('side_model.pth')
# model_top = torch.load('top_model.pth')
# model_front = torch.load('front_model.pth')
# device = 'cpu'  # AWS Free Tier uses CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_fp16_models():
    print("Loading Float16 models...")
    
    # NOTE: We use torch.load() directly because optimize_models.py 
    # saved the *entire* model object, not just the state_dict.
    try:
        model_side = torch.load('best_model_CBAM_side_new_loss_fp16.pth', map_location=device)
        model_top = torch.load('best_model_CBAM_top_new_loss_fp16.pth', map_location=device)
        model_front = torch.load('best_model_CBAM_front_new_loss_fp16.pth', map_location=device)
        
        # Ensure they are in eval mode
        model_side.eval()
        model_top.eval()
        model_front.eval()
        
        return model_side, model_top, model_front
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None

# Load them once (Global scope)
model_side, model_top, model_front = load_fp16_models()

def predict_side_view(img: Image.Image) -> Image.Image:
    """Logic for the SIDE view model using Numpy."""
    print("Running SIDE view model...")
    
    # 1. Convert PIL image to Numpy Array
    # Ensure it's RGB so we get shape (Height, Width, 3)
    input_arr = np.array(img.convert("L"))
    temp_imput = (input_arr / 255.0) * 2 - 1
    temp_imput = cv2.resize(temp_imput, (256,256))
    temp_imput = np.expand_dims(temp_imput, axis=-1)
    tensor = torch.from_numpy(temp_imput).permute(2, 0, 1).float().to(device).half()
    # model_side.eval()
    with torch.no_grad():
        preds = model_side(tensor.unsqueeze(0))
    result = np.squeeze(preds.cpu().numpy())
    final_result = ((result + 1) / 2 * 255).astype(np.uint8)
    return Image.fromarray(final_result.astype(np.uint8))

def predict_top_view(img: Image.Image) -> Image.Image:
    """Logic for the TOP view model using Numpy."""
    print("Running TOP view model...")
    
    input_arr = np.array(img.convert("L"))
    temp_imput = (input_arr / 255.0) * 2 - 1
    temp_imput = cv2.resize(temp_imput, (256,256))
    temp_imput = np.expand_dims(temp_imput, axis=-1)
    tensor = torch.from_numpy(temp_imput).permute(2, 0, 1).float().to(device).half()
    # model_top.eval()
    with torch.no_grad():
        preds = model_top(tensor.unsqueeze(0))
    result = np.squeeze(preds.cpu().numpy())
    final_result = ((result + 1) / 2 * 255).astype(np.uint8)
    return Image.fromarray(final_result.astype(np.uint8))

def predict_front_view(img: Image.Image) -> Image.Image:
    """Logic for the FRONT view model using Numpy."""
    print("Running FRONT view model...")
    
    input_arr = np.array(img.convert("L"))
    temp_imput = (input_arr / 255.0) * 2 - 1
    temp_imput = cv2.resize(temp_imput, (256,256))
    temp_imput = np.expand_dims(temp_imput, axis=-1)
    tensor = torch.from_numpy(temp_imput).permute(2, 0, 1).float().to(device).half()
    # model_side.eval()
    with torch.no_grad():
        preds = model_front(tensor.unsqueeze(0))
    result = np.squeeze(preds.cpu().numpy())
    final_result = ((result + 1) / 2 * 255).astype(np.uint8)
    return Image.fromarray(final_result.astype(np.uint8))

def run_router(input_image: Image.Image, view_type: str) -> Image.Image:
    """
    Routes the input image to the correct model based on user selection.
    """
    if view_type == 'side':
        return predict_side_view(input_image)
    elif view_type == 'top':
        return predict_top_view(input_image)
    elif view_type == 'front':
        return predict_front_view(input_image)
    else:
        # Fallback if something goes wrong
        return input_image

# =========================================
# 2. HELPER FUNCTIONS
# =========================================
def parse_image(contents):
    """Converts base64 string from Dash Upload to PIL Image."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image = Image.open(io.BytesIO(decoded))
    return image

def pil_to_b64(image):
    """Converts PIL Image to base64 string for HTML display."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

# =========================================
# 3. DASH APP LAYOUT
# =========================================
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Multi-View AI Analysis", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    
    html.Div([
        # --- VIEW SELECTOR ---
        html.Label("1. Select Camera View / Model:", style={'fontWeight': 'bold'}),
        dcc.RadioItems(
            id='view-selector',
            options=[
                {'label': ' Front View Model', 'value': 'front'},
                {'label': ' Side View Model', 'value': 'side'},
                {'label': ' Top View Model', 'value': 'top'},
            ],
            value='front', # Default selection
            inline=True,
            style={'marginBottom': '20px', 'marginTop': '10px'}
        ),

        # --- UPLOAD COMPONENT ---
        html.Label("2. Upload Image:", style={'fontWeight': 'bold'}),
        dcc.Upload(
            id='upload-image',
            children=html.Div(['Drag and Drop or ', html.A('Select an Image')]),
            style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px', 'borderStyle': 'dashed', 
                'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px 0'
            },
            multiple=False
        ),
    ], style={'maxWidth': '600px', 'margin': 'auto', 'padding': '20px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)'}),

    # --- IMAGE DISPLAY AREA ---
    html.Div([
        # Input Image
        html.Div([
            html.H4("Input"),
            html.Img(id='original-image', style={'maxWidth': '100%', 'maxHeight': '400px', 'border': '1px solid #ddd'})
        ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),

        # Output Image
        html.Div([
            html.H4(id='output-title', children="Output"), # Title changes based on model
            html.Img(id='processed-image', style={'maxWidth': '100%', 'maxHeight': '400px', 'border': '1px solid #ddd'}),
            
            # Download Button
            html.Div([
                html.Button("Download Result", id="btn-download", style={
                    'marginTop': '10px', 'backgroundColor': '#007BFF', 'color': 'white', 
                    'border': 'none', 'padding': '10px 20px', 'cursor': 'pointer'
                }),
                dcc.Download(id="download-component")
            ], id="download-container", style={'display': 'none'})
            
        ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'})
    
    ], style={'margin': '20px', 'textAlign': 'center'}),
])

# =========================================
# 4. CALLBACKS
# =========================================

# Callback to process the image
@callback(
    Output('original-image', 'src'),
    Output('processed-image', 'src'),
    Output('download-container', 'style'),
    Output('output-title', 'children'),
    Input('upload-image', 'contents'),
    Input('view-selector', 'value') # Listen to selector changes
)
def update_output(contents, view_type):
    # If no image is uploaded yet, do nothing
    if contents is None:
        return no_update, no_update, {'display': 'none'}, "Output"
    
    # 1. Parse uploaded image
    input_img = parse_image(contents)
    
    # 2. Run the routing logic based on view_type
    output_img = run_router(input_img, view_type)
    
    # 3. Convert back to displayable format
    output_b64 = pil_to_b64(output_img)
    
    # 4. Update UI
    download_style = {'display': 'block', 'marginTop': '10px'}
    title_text = f"Prediction ({view_type.capitalize()} Model)"
    
    return contents, output_b64, download_style, title_text

# Callback to download the result
@callback(
    Output("download-component", "data"),
    Input("btn-download", "n_clicks"),
    State("processed-image", "src"),
    State("view-selector", "value"),
    prevent_initial_call=True
)
def download_image(n_clicks, src, view_type):
    if not src:
        return no_update

    content_string = src.split(',')[1]
    decoded = base64.b64decode(content_string)
    
    filename = f"processed_{view_type}_view.png"
    return dcc.send_bytes(decoded, filename)

# =========================================
# 5. RUN SERVER
# =========================================
if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)