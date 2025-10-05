import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("test_Vector.csv")


features = [
    'departure_delay_norm', 
    'ground_time_pressure_norm', 
    'pax_load_ratio_norm', 
    'transfer_ratio_norm', 
    'child_ratio_norm', 
    'special_req_count_norm'
]
target = 'difficulty_class'

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

st.title("✈️ Flight Difficulty Predictor")
st.write("Predict flight difficulty based on selected flight and date, or manually adjust parameters.")
st.sidebar.header("🔍 Flight Lookup")

flight_numbers = df['flight_number'].unique()
selected_flight = st.sidebar.selectbox("Select Flight Number", sorted(flight_numbers))

available_dates = df[df['flight_number'] == selected_flight]['scheduled_departure_date_local'].unique()
selected_date = st.sidebar.selectbox("Select Date", sorted(available_dates))

selected_row = df[(df['flight_number'] == selected_flight) & 
                  (df['scheduled_departure_date_local'] == selected_date)]

if not selected_row.empty:
    st.success(f"Flight {selected_flight} found on {selected_date}")
    flight_data = selected_row.iloc[0]
else:
    st.error("No data found for this flight and date.")
    st.stop()

st.subheader("📋 Flight Summary")
st.write(selected_row[['scheduled_arrival_station_code', 'fleet_type', 'total_seats',
                       'departure_delay', 'ground_time_pressure', 'total_pax']])

st.subheader("🎚️ Adjust Normalized Parameters")

input_data = {}
for feature in features:
    input_data[feature] = st.slider(
        feature.replace("_norm", "").replace("_", " ").title(),
        0.0, 1.0, float(flight_data[feature])
    )

input_df = pd.DataFrame([input_data])

if st.button("🚀 Predict Difficulty"):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    st.subheader(f"Predicted Difficulty: **{pred}**")
    st.write(f"Confidence: {max(proba) * 100:.2f}%")

    
    prob_df = pd.DataFrame({
        "Difficulty": model.classes_,
        "Probability": proba
    }).sort_values(by="Probability", ascending=False)
    st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}))


if st.checkbox("Show Feature Importance"):
    st.write("### 🔍 Feature Importance")
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    st.dataframe(importance_df)
    st.bar_chart(importance_df.set_index("Feature"))

st.caption("Developed by Team Vector ✈️ – SkyHack 3.0: United Airlines")
