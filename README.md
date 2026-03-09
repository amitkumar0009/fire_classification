# 🔥 Fire Type Prediction using MODIS Data

This project is a **Machine Learning based web application** that predicts the **type of fire** using satellite data from the **MODIS (Moderate Resolution Imaging Spectroradiometer)** dataset.

The application is built using **Python, Scikit-Learn, and Streamlit** and allows users to input satellite and environmental parameters to predict the fire type.

---

## 🌐 Live Application

Access the deployed application here:

### Demo : https://fireclassification-psf8ut7ykzfjwptnfxlauc.streamlit.app/

---

## 📊 Features

- Predict fire type using a trained Machine Learning model
- Interactive web interface built with **Streamlit**
- Accepts multiple satellite data parameters
- Uses saved **label encoders** for categorical features
- Real-time prediction

---

## 🧠 Machine Learning Model

The model was trained using **MODIS fire detection dataset** with the following features:

- Latitude
- Longitude
- Brightness
- Scan
- Track
- Acquisition Time
- Satellite
- Instrument
- Confidence
- Brightness T31
- Fire Radiative Power (FRP)
- Day/Night





---

## 🖥 Application Workflow

1. User enters satellite data parameters
2. Categorical features are encoded using saved encoders
3. Data is passed to the trained ML model
4. Model predicts the **fire type**
5. Result is displayed instantly in the Streamlit interface

---

## 🛠 Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Joblib
- Pickle

---

## 📌 Future Improvements

- Improve prediction accuracy with larger datasets
- Add visualization dashboards
- Integrate real-time satellite data
- Deploy using Docker or cloud platforms

---

## 👨‍💻 Author

Amit Kumar  
B.Tech Information Technology  
Netaji Subhas University of Technology (NSUT)

---

⭐ If you found this project helpful, consider giving the repository a star.
