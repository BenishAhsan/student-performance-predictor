from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI()

# CORS — frontend (HTML file) ko allow karta hai API call karne ke liye
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = joblib.load("best_model.pkl")

# Input schema with validation
class StudentInput(BaseModel):
    study_hours_per_day: float = Field(ge=0, le=10, description="Daily study hours (0–10)")
    attendance_percentage: float = Field(ge=0, le=100, description="Attendance % (0–100)")
    mental_health_rating: int = Field(ge=1, le=10, description="Mental health rating (1–10, 10 is best)")
    sleep_hours: float = Field(ge=3, le=10, description="Daily sleep hours (3–10)")
    social_media_hours: float = Field(ge=0, le=8, description="Daily social media hours (0–8)")
    netflix_hours: float = Field(ge=0, le=7, description="Daily Netflix hours (0–7)")
    exercise_frequency: int = Field(ge=0, le=6, description="Exercise days per week (0–6)")
    part_time_job: int = Field(ge=0, le=1, description="Part-time job: 0 = No, 1 = Yes")
    gender: int = Field(ge=0, le=1, description="Gender (LabelEncoded: 0 or 1)")
    diet_quality: int = Field(ge=0, le=2, description="Diet quality (0=Poor, 1=Fair, 2=Good)")
    internet_quality: int = Field(ge=0, le=2, description="Internet quality (0=Poor, 1=Average, 2=Good)")
    extracurricular_participation: int = Field(ge=0, le=1, description="Extracurricular: 0 = No, 1 = Yes")

@app.get("/")
def home():
    return {"message": "Student Performance Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: StudentInput):
    input_data = [[
        data.study_hours_per_day,
        data.attendance_percentage,
        data.mental_health_rating,
        data.sleep_hours,
        data.social_media_hours,
        data.netflix_hours,
        data.exercise_frequency,
        data.part_time_job,
        data.gender,
        data.diet_quality,
        data.internet_quality,
        data.extracurricular_participation
    ]]

    prediction = model.predict(input_data)[0]
    predicted_score = round(float(np.clip(prediction, 0, 100)), 2)

    return {
        "predicted_exam_score": predicted_score,
        "model_used": type(model).__name__
    }

# Run with: uvicorn app:app --reload
