import os
from sqlalchemy import create_engine, Column, String, DateTime, JSON, MetaData, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json
import pandas as pd
from typing import Optional, Dict, Any

# Get the current directory and create a db directory if it doesn't exist
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, 'data')
os.makedirs(db_dir, exist_ok=True)
db_path = os.path.join(db_dir, 'eda.db')

# Database Configuration (using SQLite)
DATABASE_URL = f"sqlite:///{db_path}"

# SQLAlchemy setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Models
class Upload(Base):
    __tablename__ = "uploads"

    id = Column(String, primary_key=True)
    filename = Column(String)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    dataframe_json = Column(JSON)

class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True)
    query = Column(String)
    response = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Create tables
def init_db():
    Base.metadata.create_all(bind=engine)

# Database operations
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def store_dataframe(file_id: str, filename: str, df: pd.DataFrame) -> None:
    """Store DataFrame as JSON in the database"""
    db = SessionLocal()
    try:
        upload = Upload(
            id=file_id,
            filename=filename,
            dataframe_json=json.loads(df.to_json(orient='split'))
        )
        db.add(upload)
        db.commit()
    finally:
        db.close()

def load_dataframe(file_id: str) -> Optional[pd.DataFrame]:
    """Load DataFrame from JSON stored in database"""
    db = SessionLocal()
    try:
        upload = db.query(Upload).filter(Upload.id == file_id).first()
        if upload:
            return pd.read_json(json.dumps(upload.dataframe_json), orient='split')
        return None
    finally:
        db.close()

def store_session(session_id: str, query: str, response: Dict[str, Any]) -> None:
    """Store chat session in database"""
    db = SessionLocal()
    try:
        session = Session(
            id=session_id,
            query=query,
            response=response
        )
        db.add(session)
        db.commit()
    finally:
        db.close()

def get_session_history() -> list:
    """Get all chat sessions"""
    db = SessionLocal()
    try:
        sessions = db.query(Session).order_by(Session.timestamp.desc()).all()
        return [
            {
                "id": s.id,
                "query": s.query,
                "response": s.response,
                "timestamp": s.timestamp.isoformat()
            }
            for s in sessions
        ]
    finally:
        db.close()
