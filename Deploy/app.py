import base64
import io
from dash import Dash, dcc, html, Input, Output, State, callback, no_update
from PIL import Image, ImageOps

# =========================================
# 1. MODEL DEFINITION
# =========================================
def run_ai_model(input_image: Image.Image) -> Image.Image:
    """
    This is your 'Model' placeholder. 
    Replace the logic inside this function to change the model.
    
    Args:
        input_image (PIL.Image): The uploaded image.
    Returns:
        PIL.Image: The processed image.
    """
    # --- START OF MODEL LOGIC ---
    # Example: Simple image translation (Invert colors + Grayscale)
    # In a real app, you would pass 'input_image' to PyTorch/TensorFlow here.
    processed_image = ImageOps.invert(input_image.convert("RGB"))
    processed_image = processed_image.convert("L")  # Convert to grayscale
    # --- END OF MODEL LOGIC ---
    
    return processed_image

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
    html.H1("AI Image Translation App", style={'textAlign': 'center'}),
    
    # Upload Component
    html.Div([
        dcc.Upload(
            id='upload-image',
            children=html.Div(['Drag and Drop or ', html.A('Select an Image')]),
            style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px', 'borderStyle': 'dashed', 
                'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
            },
            multiple=False
        ),
    ], style={'maxWidth': '600px', 'margin': 'auto'}),

    # Image Display Area (Side by Side)
    html.Div([
        # Input Image
        html.Div([
            html.H4("Input"),
            html.Img(id='original-image', style={'maxWidth': '100%', 'maxHeight': '400px'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        # Output Image
        html.Div([
            html.H4("Prediction (Translated)"),
            html.Img(id='processed-image', style={'maxWidth': '100%', 'maxHeight': '400px'}),
            
            # Download Button (Hidden initially)
            html.Div([
                html.Button("Download Result", id="btn-download", style={'marginTop': '10px'}),
                dcc.Download(id="download-component")
            ], id="download-container", style={'display': 'none'})
            
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'})
    
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
    Input('upload-image', 'contents')
)
def update_output(contents):
    if contents is None:
        return no_update, no_update, {'display': 'none'}
    
    # 1. Parse uploaded image
    input_img = parse_image(contents)
    
    # 2. Run the model
    output_img = run_ai_model(input_img)
    
    # 3. Convert back to displayable format
    output_b64 = pil_to_b64(output_img)
    
    # Show download button
    download_style = {'display': 'block', 'marginTop': '10px'}
    
    return contents, output_b64, download_style

# Callback to download the result
@callback(
    Output("download-component", "data"),
    Input("btn-download", "n_clicks"),
    State("processed-image", "src"),
    prevent_initial_call=True
)
def download_image(n_clicks, src):
    if not src:
        return no_update

    # Extract base64 part from the data URL
    content_string = src.split(',')[1]
    decoded = base64.b64decode(content_string)
    
    # Trigger download
    return dcc.send_bytes(decoded, "processed_result.png")

# =========================================
# 5. RUN SERVER
# =========================================
if __name__ == '__main__':
    app.run(debug=True)