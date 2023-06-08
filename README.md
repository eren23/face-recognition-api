# Face Recognition Service using FastAPI

This is a simple Face Recognition Service implemented using FastAPI, MongoDB or SQLite, and FaceNet model for face recognition.

## Getting Started

### Prerequisites

- Python 3.9 or later.
- A MongoDB instance or SQLite database.
- Environment variable configuration file `.env`.

### Installing

Clone the repository:

git clone https://github.com/eren23/face-recognition-api.git
cd repository

Install dependencies:

pip install -r requirements.txtv

## Running the Application

The application can be started with the following command:

uvicorn main:app --reload

The `--reload` flag enables hot reloading which means the server will automatically update whenever you make changes to your code.

## Configuration

The service can use either MongoDB or SQLite for storing embeddings, this can be configured using the `DB_CHOICE` environment variable in the `.env` file:

DB_CHOICE=mongo

or

DB_CHOICE=sqlite

The connection string for the MongoDB instance is taken from the `MONGO_CONNECTION_STRING` environment variable in the `.env` file:

MONGO_CONNECTION_STRING=yourconnectionstring

Remember to replace the connection string with your own.

## Endpoints

The service exposes two endpoints:

- `POST /add_face/{user_id}`: Add a new face to a user. The user is identified by `user_id` and the face is passed in the form of an uploaded image file. The service will extract the face embedding from the image and store it in the database associated with the user.

- `POST /identify_face/`: Identify a face in an uploaded image. The service will extract the face embedding from the image and compare it with the embeddings stored in the database. If a match is found, it returns the `user_id` associated with the match.
