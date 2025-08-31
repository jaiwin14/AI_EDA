import plotly.graph_objects as go
import plotly.express as px
import base64
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

def export_plot_as_image(fig, format="png", width=800, height=600):
    """
    Export a plotly figure as a base64 encoded image with error handling
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure or plotly.express.Figure
        The plotly figure to export
    format : str, optional
        Image format (png, jpg, svg, pdf). Default is "png"
    width : int, optional
        Image width in pixels. Default is 800
    height : int, optional
        Image height in pixels. Default is 600
    
    Returns:
    --------
    str or None
        Base64 encoded image string, or None if export fails
    """
    try:
        # Update figure layout with dimensions
        fig.update_layout(
            width=width,
            height=height,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Try to export using kaleido
        buffer = BytesIO()
        fig.write_image(buffer, format=format, engine="kaleido")
        buffer.seek(0)
        
        # Encode to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        return image_base64
        
    except ImportError as e:
        logger.error(f"Kaleido not available: {e}")
        return _fallback_export(fig, format, width, height)
    except Exception as e:
        logger.error(f"Error exporting plot: {e}")
        return _fallback_export(fig, format, width, height)

def _fallback_export(fig, format="png", width=800, height=600):
    """
    Fallback export method when kaleido is not available
    """
    try:
        # Try using matplotlib as fallback
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Convert plotly to matplotlib (basic conversion)
        fig_bytes = fig.to_image(format=format, width=width, height=height)
        return base64.b64encode(fig_bytes).decode()
        
    except Exception as e:
        logger.error(f"Fallback export also failed: {e}")
        # Return a simple text representation
        return _create_text_fallback(fig)

def _create_text_fallback(fig):
    """
    Create a simple text representation when image export fails
    """
    try:
        # Create a simple text summary of the plot
        if hasattr(fig, 'data') and fig.data:
            plot_type = type(fig.data[0]).__name__
            return f"Plot type: {plot_type} - Image export not available. Please install kaleido: pip install kaleido"
        else:
            return "Plot created successfully but image export not available. Please install kaleido: pip install kaleido"
    except:
        return "Plot created successfully but image export not available. Please install kaleido: pip install kaleido"

def create_error_plot(error_message, width=800, height=400):
    """
    Create a plot showing an error message
    
    Parameters:
    -----------
    error_message : str
        The error message to display
    width : int, optional
        Plot width. Default is 800
    height : int, optional
        Plot height. Default is 400
    
    Returns:
    --------
    str
        Base64 encoded error plot
    """
    fig = go.Figure()
    
    fig.add_annotation(
        text=error_message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        xanchor="center", yanchor="middle",
        showarrow=False,
        font=dict(size=16, color="red"),
        bgcolor="rgba(255, 200, 200, 0.8)",
        bordercolor="red",
        borderwidth=2
    )
    
    fig.update_layout(
        width=width,
        height=height,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor="white",
        title="Error in Analysis"
    )
    
    return export_plot_as_image(fig, width=width, height=height)
