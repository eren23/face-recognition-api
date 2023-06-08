# systems stuff
import io
import os
from dotenv import load_dotenv

# FastAPI stuff
from fastapi import FastAPI, UploadFile, HTTPException

# PyTorch stuff
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# MongoDB stuff
from pymongo import MongoClient

# SQLite stuff
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, String, ARRAY, Float
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Text
from sqlalchemy import ForeignKey, Integer, Column
from sqlalchemy.orm import relationship

# Other stuff
from scipy.spatial.distance import cosine

# Load environment variables
load_dotenv()

# Get database choice from environment variable
db_choice = os.getenv("DB_CHOICE")

# Use CPU
device = torch.device('cpu')

# Initialize FaceNet model and MTCNN detector
mtcnn = MTCNN(device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Initialize FastAPI
app = FastAPI()

print(db_choice, "Db choice")

if db_choice == 'mongo':
    client = MongoClient(os.getenv("MONGO_CONNECTION_STRING"))
    db = client['face_embeddings']
    users = db['users']
elif db_choice == 'sqlite':
    engine = create_engine('sqlite:///./face_embeddings.db')
    SessionLocal = sessionmaker(bind=engine)
    Base = declarative_base()

    class User(Base):
        __tablename__ = "users"
    
        id = Column(Integer, primary_key=True, index=True)
        user_id = Column(String, unique=True, index=True)
        embeddings = relationship("Embedding", back_populates="owner")


    class Embedding(Base):
        __tablename__ = "embeddings"

        id = Column(Integer, primary_key=True, index=True)
        embedding = Column(String)
        owner_id = Column(Integer, ForeignKey("users.id"))

        owner = relationship("User", back_populates="embeddings")

    Base.metadata.create_all(bind=engine)

else:
    raise ValueError("Invalid database choice")


def preprocess_image(image: Image.Image):
    
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Detect faces
    face, _ = mtcnn(img_array, return_prob=True)
    if face is not None:
        return face
    else:
        raise ValueError("No face detected in the image")

def generate_embedding(face: torch.Tensor):
    # Genrate face embeddings
    embedding = facenet(face.unsqueeze(0)).detach().numpy()[0]
    return embedding

if db_choice == 'mongo':
    @app.post("/add_face/{user_id}")
    async def add_face(user_id: str, file: UploadFile):
        
        # Check if user_id exists, if not create, without this check the app will crash
        user = users.find_one({'user_id': user_id})
        if not user:
            user = {"user_id": user_id, "embeddings": []}
            users.insert_one(user)

        # Load image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Preprocess the image
        try:
            face = preprocess_image(image)
        except ValueError:
            raise HTTPException(status_code=400, detail="No face detected in the image")

        # Generate embedding for the face
        embedding = generate_embedding(face)

        # Store the embedding in the MongoDB collection, I currently push the embedding to an array, you can use a different method to keep the last embedding
        users.update_one(
            {'user_id': user_id},
            {'$push': {'embeddings': embedding.tolist()}},
        )

        return {"success": True}

elif db_choice == 'sqlite':
    @app.post("/add_face/{user_id}")
    async def add_face(user_id: str, file: UploadFile):
        
        db = SessionLocal()

        # Check if user_id exists, if not create it, same story as above
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            # Create new user
            user = User(user_id=user_id)
            db.add(user)
            db.commit()

        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        try:
            face = preprocess_image(image)
        except ValueError:
            db.close()
            raise HTTPException(status_code=400, detail="No face detected in the image")

        embedding = generate_embedding(face)

        # Store the embedding in the SQLite database
        # Convert the numpy array to a string
        embedding_str = ' '.join(map(str, embedding.tolist()))

        # Create new Embedding object
        new_embedding = Embedding(embedding=embedding_str, owner_id=user.id)
        
        db.add(new_embedding)

        db.commit()

        db.close()

        return {"success": True}
    
@app.post("/identify_face/")
async def identify_face(file: UploadFile):
    
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    try:
        face = preprocess_image(image)
    except ValueError:
        raise HTTPException(status_code=400, detail="No face detected in the image")

    embedding = generate_embedding(face)

    # Compare the embedding with embeddings in the database
    if db_choice == 'mongo':
        for user in users.find():
            for saved_embedding in user['embeddings']:
                score = cosine(np.array(saved_embedding), embedding)
                if score <= 0.4: # consider it a match if score is <= 0.4
                    return {"match": True, "user_id": user['user_id']}
                
    elif db_choice == 'sqlite':
        db = SessionLocal()
        for user in db.query(User).all():
            for saved_embedding in user.embeddings:
                # Convert the string back to a list of floats and then to numpy array
                saved_embedding_array = np.array(list(map(float, saved_embedding.embedding.split(' '))), dtype=np.float32)
                score = cosine(saved_embedding_array, embedding)
                if score <= 0.4: 
                    return {"match": True, "user_id": user.user_id}
    return {"match": False}