@echo off
echo Starting Addis Ride Demand Forecast App...
echo.
cd app
call ..\venv\Scripts\activate
streamlit run app.py
