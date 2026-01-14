"""ETF Portfolio Tool - Main Application Entry Point."""

import dash
import dash_bootstrap_components as dbc
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from ui.callbacks import register_callbacks
from ui.layouts import create_app_layout

# Initialize the Dash app with Bloomberg-style dark theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],  # Dark base theme
    title="ETF Portfolio Tool | Bloomberg Style",
    suppress_callback_exceptions=True,
)

# Set the layout
app.layout = create_app_layout()

# Register callbacks
register_callbacks(app)

# Expose server for WSGI
server = app.server

if __name__ == "__main__":
    print("=" * 50)
    print("ETF Portfolio Tool")
    print("=" * 50)
    print("Starting server at http://localhost:8050")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    app.run(
        debug=True,
        host="127.0.0.1",
        port=8050,
    )
