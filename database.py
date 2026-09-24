from sqlalchemy import create_engine, Column, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
import uuid

# Using SQLite for local enterprise testing. Easily swaps to PostgreSQL in Docker.
SQLALCHEMY_DATABASE_URL = "sqlite:///./phytovision.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Organization(Base):
    """B2B Tenant (e.g., Drone Corp, Insurance MNC)"""
    __tablename__ = 'organizations'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    api_key = Column(String, unique=True, nullable=False)
    
    farms = relationship("Farm", back_populates="organization")

class Farm(Base):
    """Geographic grouping of crops"""
    __tablename__ = 'farms'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String, ForeignKey('organizations.id'))
    name = Column(String, nullable=False)
    
    organization = relationship("Organization", back_populates="farms")
    diagnostic_events = relationship("DiagnosticEvent", back_populates="farm")

class DiagnosticEvent(Base):
    """The core operational record for analytics and reporting"""
    __tablename__ = 'diagnostic_events'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    farm_id = Column(String, ForeignKey('farms.id'))
    
    # AI Predictions
    disease_predicted = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=False)
    severity_percentage = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)
    
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    farm = relationship("Farm", back_populates="diagnostic_events")

# Generate the tables
Base.metadata.create_all(bind=engine)