# QuickRoom - Flask, Tailwind, MongoDB

This is a simple Flask application that demonstrates API functionality, HTML templating with Tailwind CSS (via CDN), and MongoDB integration for CRUD operations.

## Project Structure

```
/
├── app.py                  # Main Flask application
├── config.py               # Configuration loader (for .env)
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Main HTML page with Tailwind CSS and JS for API interaction
├── static/                 # For other static assets (optional)
├── .env                    # Local environment variables (you need to create this from .env.example or manually)
└── README.md               # This file
```

## Setup Instructions

1.  **Clone the repository (if applicable) or create the files as shown.**

2.  **Create a Python virtual environment:**
    ```bash
    python -m venv venv
    ```
    Activate the virtual environment:
    *   Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    *   macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a file named `.env` in the root directory of the project.
    Add your MongoDB connection string to this file:
    ```env
    MONGO_URI="your_mongodb_atlas_connection_string_or_local_uri"
    FLASK_DEBUG=True
    ```
    Replace `"your_mongodb_atlas_connection_string_or_local_uri"` with your actual MongoDB URI. 
    For example, a local MongoDB URI might look like `mongodb://localhost:27017/quick_room_db` (ensure `quick_room_db` or your chosen database name is part of the URI if you want `get_default_database()` to work seamlessly, otherwise the app defaults to `quick_room_db`).

5.  **Ensure MongoDB is running and accessible.**

## Running the Application

Once the setup is complete, you can run the Flask application:

```bash
python app.py
```

The application will typically be available at `http://127.0.0.1:5000/` or `http://0.0.0.0:5000/`.

## API Endpoints

The application provides the following API endpoints for managing items:

*   `GET /api/items`: Retrieves all items.
*   `POST /api/items`: Adds a new item. Requires a JSON body with `name` and `description`.
    *   Example: `{"name": "New Item", "description": "This is a new item."}`
*   `GET /api/items/<item_id>`: Retrieves a specific item by its ID.
*   `PUT /api/items/<item_id>`: Updates an existing item. Requires a JSON body with `name` and/or `description`.
*   `DELETE /api/items/<item_id>`: Deletes an item by its ID.

The `index.html` page provides a simple UI to interact with these endpoints.

## Tailwind CSS

Tailwind CSS is included via a CDN link in `templates/index.html`. No local build step for Tailwind is required with this setup. 