# 🎓 Student Performance Prediction

Predicts a student's exam score based on lifestyle and academic habit data — 
not just grades, but sleep, mental health, social media usage, and more.

## 🧩 Problem Statement
Can daily habits predict academic performance? This project explores that 
question using real behavioral data across 1000 students.

## 📊 Dataset
- **1000 students**, 16 features
- **Target:** `exam_score` (continuous — regression task)
- **Train/Test Split:** 727 / 182 (80/20)

## 🔢 Features Used
| Feature | Type |
|---|---|
| study_hours_per_day | Numeric |
| attendance_percentage | Numeric |
| mental_health_rating | Numeric |
| sleep_hours | Numeric |
| social_media_hours | Numeric |
| netflix_hours | Numeric |
| exercise_frequency | Numeric |
| part_time_job | Categorical |
| diet_quality | Categorical |
| internet_quality | Categorical |
| extracurricular_participation | Categorical |
| parental_education_level | Categorical |

## ⚙️ ML Pipeline
- Exploratory Data Analysis (correlation heatmap, scatter plots, distributions)
- Label Encoding for categorical features
- Models trained: Linear Regression, Decision Tree, Random Forest
- Hyperparameter tuning: GridSearchCV with 5-fold cross validation
- Best model auto-selected by lowest RMSE

## 📈 Results
| Model | Result |
|---|---|
| Linear Regression | ✅ Best — RMSE: 5.5034 |
| Decision Tree | GridSearchCV tuned |
| Random Forest | GridSearchCV tuned |

**Sample Predictions vs Actual:**
| Predicted | Actual |
|---|---|
| 61.96 | 59.6 |
| 51.31 | 43.3 |
| 70.90 | 68.4 |
| 71.11 | 75.1 |
| 87.24 | 87.9 |

## 🚀 Tech Stack
`Python` `pandas` `scikit-learn` `FastAPI` `uvicorn` `joblib` `AWS EC2`

## 🖥️ Run Locally
```bash
git clone https://github.com/yourusername/student-performance-prediction
cd student-performance-prediction
pip install fastapi uvicorn scikit-learn pandas joblib
uvicorn app:app --reload
# Open http://127.0.0.1:8000/docs
```

## 📸 Screenshots
<img width="1277" height="700" alt="Launched Successfully" src="https://github.com/user-attachments/assets/fb92b954-daa7-4cf2-bcdb-b14197c0eb72" />
<img width="1279" height="764" alt="Beanstalk Responses" src="https://github.com/user-attachments/assets/bd53f65b-8636-445f-b572-a52b11e9e82f" />

