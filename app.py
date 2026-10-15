import time
import threading
import queue
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import serial
from serial.tools import list_ports
import joblib
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh

# ---------------- Page and files ----------------
st.set_page_config(page_title="Smart Room ML Dashboard", page_icon="🏠", layout="wide")
APP_DIR = Path(__file__).resolve().parent
DATA_FILE = APP_DIR / "smart_room_dataset.csv"
MODEL_FILE = APP_DIR / "smart_room_model.joblib"
FEATURES = ["motion", "distance_cm", "light_adc"]
LABELS = ["EMPTY", "OCCUPIED"]

st.title("🏠 Smart Room Occupancy — ML Dashboard")
st.caption("ESP32 • PIR • HC-SR04 • LDR • DHT11 • Decision Tree")
st.info(
    "This version trains and runs ML on the computer using live ESP32 serial data. "
    "It does not yet send predictions back to the ESP32 or switch mains appliances."
)

# Refresh UI periodically so queued serial readings appear without clicking.
st_autorefresh(interval=1500, key="dashboard_refresh")

def parse_serial_line(line):
    """Parse short test firmware and longer full firmware CSV lines."""
    parts = [part.strip() for part in line.strip().split(",")]
    if len(parts) < 4:
        return None
    try:
        row = {
            "device_time_ms": float(parts[0]),
            "motion": int(float(parts[1])),
            "distance_cm": float(parts[2]),
            "light_adc": int(float(parts[3])),
            "temperature_c": None,
            "humidity_pct": None,
            "device_status": None,
            "received_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        if row["motion"] not in (0, 1):
            return None
        if len(parts) >= 8:
            row["temperature_c"] = float(parts[4]) if parts[4].lower() not in ("nan", "") else None
            row["humidity_pct"] = float(parts[5]) if parts[5].lower() not in ("nan", "") else None
            row["device_status"] = parts[7]
        return row
    except (ValueError, IndexError):
        # Ignore headers, boot messages, and malformed lines.
        return None

def serial_reader(port, baud, output_queue, stop_event):
    connection = None
    try:
        connection = serial.Serial(port, baud, timeout=1)
        time.sleep(2)  # Many ESP32 boards reset when serial opens.
        while not stop_event.is_set():
            line = connection.readline().decode("utf-8", errors="ignore").strip()
            if line:
                row = parse_serial_line(line)
                if row is not None:
                    output_queue.put(row)
    except Exception as exc:
        output_queue.put({"error": str(exc)})
    finally:
        if connection is not None and connection.is_open:
            connection.close()

# ---------------- Session state ----------------
defaults = {
    "serial_queue": queue.Queue(),
    "stop_event": threading.Event(),
    "serial_thread": None,
    "live_rows": [],
    "last_reading": None,
    "serial_error": "",
    "model": None,
    "model_metrics": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

if "dataset" not in st.session_state:
    try:
        st.session_state.dataset = pd.read_csv(DATA_FILE)
    except Exception:
        st.session_state.dataset = pd.DataFrame(columns=FEATURES + ["label", "timestamp"])

if st.session_state.model is None and MODEL_FILE.exists():
    try:
        st.session_state.model = joblib.load(MODEL_FILE)
    except Exception:
        st.session_state.model = None

def save_dataset():
    st.session_state.dataset.to_csv(DATA_FILE, index=False)

# ---------------- Sidebar connection ----------------
with st.sidebar:
    st.header("ESP32 connection")
    detected_ports = [item.device for item in list_ports.comports()]
    port_options = detected_ports or ["COM5", "/dev/ttyUSB0", "/dev/ttyACM0"]
    selected_port = st.selectbox("Serial port", port_options)
    selected_baud = st.selectbox("Baud rate", [115200, 9600, 57600], index=0)

    thread = st.session_state.serial_thread
    connected = thread is not None and thread.is_alive()

    if not connected:
        if st.button("Connect to ESP32", use_container_width=True):
            st.session_state.stop_event = threading.Event()
            st.session_state.serial_error = ""
            new_thread = threading.Thread(
                target=serial_reader,
                args=(selected_port, selected_baud, st.session_state.serial_queue, st.session_state.stop_event),
                daemon=True,
            )
            st.session_state.serial_thread = new_thread
            new_thread.start()
            st.rerun()
    else:
        st.success(f"Serial reader active: {selected_port}")
        if st.button("Disconnect", use_container_width=True):
            st.session_state.stop_event.set()
            st.session_state.serial_thread = None
            st.rerun()

    st.divider()
    st.metric("Labeled samples", len(st.session_state.dataset))
    st.download_button(
        "Download dataset CSV",
        st.session_state.dataset.to_csv(index=False).encode("utf-8"),
        file_name="smart_room_dataset.csv",
        mime="text/csv",
        use_container_width=True,
    )
    uploaded_csv = st.file_uploader("Import labeled CSV", type=["csv"])
    if uploaded_csv is not None:
        try:
            incoming = pd.read_csv(uploaded_csv)
            if all(col in incoming.columns for col in FEATURES + ["label"]):
                incoming["label"] = incoming["label"].astype(str).str.upper()
                incoming = incoming[incoming["label"].isin(LABELS)]
                st.session_state.dataset = pd.concat(
                    [st.session_state.dataset, incoming], ignore_index=True
                ).drop_duplicates()
                save_dataset()
                st.success("CSV imported.")
            else:
                st.error("CSV columns required: motion, distance_cm, light_adc, label.")
        except Exception as exc:
            st.error(f"Could not import CSV: {exc}")

# Process serial queue
while True:
    try:
        item = st.session_state.serial_queue.get_nowait()
    except queue.Empty:
        break
    if "error" in item:
        st.session_state.serial_error = item["error"]
    else:
        st.session_state.last_reading = item
        st.session_state.live_rows.append(item)
st.session_state.live_rows = st.session_state.live_rows[-500:]

if st.session_state.serial_error:
    st.error("Serial connection error: " + st.session_state.serial_error)

last = st.session_state.last_reading
model = st.session_state.model
prediction, confidence = "Waiting for data", None
if last:
    if model is not None and last["distance_cm"] >= 0:
        live_x = pd.DataFrame(
            [[last["motion"], last["distance_cm"], last["light_adc"]]],
            columns=FEATURES,
        )
        prediction = str(model.predict(live_x)[0])
        if hasattr(model, "predict_proba"):
            confidence = float(max(model.predict_proba(live_x)[0]))
    else:
        prediction = "Model not trained"

top1, top2, top3, top4 = st.columns(4)
top1.metric("Connection", "Receiving" if connected else "Not connected")
top2.metric("PIR motion", "Detected" if last and last["motion"] else ("No motion" if last else "—"))
top3.metric("ML prediction", prediction)
top4.metric("Model confidence", f"{confidence:.0%}" if confidence is not None else "—")

live_tab, data_tab, model_tab, info_tab = st.tabs(
    ["📡 Live Monitor", "🧪 Collect Training Data", "🧠 Train & Evaluate ML", "ℹ️ Project Info"]
)

with live_tab:
    st.subheader("Live sensor readings")
    if last:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Distance", f'{last["distance_cm"]:.1f} cm' if last["distance_cm"] >= 0 else "No echo")
        c2.metric("LDR ADC", last["light_adc"])
        c3.metric("Temperature", f'{last["temperature_c"]} °C' if last["temperature_c"] is not None else "Not in current firmware")
        c4.metric("Humidity", f'{last["humidity_pct"]}%' if last["humidity_pct"] is not None else "Not in current firmware")
        c5.metric("Device status", last["device_status"] or "—")
        st.caption(f'Last reading: {last["received_at"]}')
    else:
        st.warning("No readings yet. Close Arduino Serial Monitor, select the ESP32 COM port in the sidebar, then click Connect to ESP32.")

    if st.session_state.live_rows:
        history = pd.DataFrame(st.session_state.live_rows)
        st.line_chart(history.set_index("received_at")[["distance_cm", "light_adc"]].tail(100))
        st.dataframe(history.tail(20), use_container_width=True, hide_index=True)

with data_tab:
    st.subheader("Collect labeled training examples")
    st.write("Label the actual room state—not the model's prediction.")
    selected_label = st.radio("Actual room state", LABELS, horizontal=True)
    n_samples = st.slider("Recent readings to save", 1, 20, 5)
    if st.button("Save latest readings with selected label", type="primary"):
        valid_rows = [r for r in st.session_state.live_rows if r["distance_cm"] >= 0]
        selected_rows = valid_rows[-n_samples:]
        if not selected_rows:
            st.warning("No valid distance readings available yet.")
        else:
            additions = pd.DataFrame([
                {
                    "motion": r["motion"],
                    "distance_cm": r["distance_cm"],
                    "light_adc": r["light_adc"],
                    "label": selected_label,
                    "timestamp": r["received_at"],
                } for r in selected_rows
            ])
            st.session_state.dataset = pd.concat([st.session_state.dataset, additions], ignore_index=True)
            save_dataset()
            st.success(f"Saved {len(additions)} samples as {selected_label}.")

    if len(st.session_state.dataset):
        st.bar_chart(st.session_state.dataset["label"].value_counts())
        st.dataframe(st.session_state.dataset.tail(30), use_container_width=True, hide_index=True)
    else:
        st.info("Collect both EMPTY and OCCUPIED samples under different conditions.")

    if st.button("Clear training dataset"):
        st.session_state.dataset = pd.DataFrame(columns=FEATURES + ["label", "timestamp"])
        save_dataset()
        st.warning("Dataset cleared.")

with model_tab:
    st.subheader("Train a Decision Tree model")
    dataset = st.session_state.dataset.dropna(subset=FEATURES + ["label"]).copy()
    dataset = dataset[dataset["label"].isin(LABELS)]
    st.write(f"Valid samples: {len(dataset)}; class counts: {dataset['label'].value_counts().to_dict() if len(dataset) else {}}")
    if st.button("Train Decision Tree", type="primary"):
        if len(dataset) < 20 or dataset["label"].nunique() < 2 or dataset["label"].value_counts().min() < 2:
            st.error("Collect at least 20 valid samples with both EMPTY and OCCUPIED labels. Aim for 100+ per class across varied conditions.")
        else:
            x = dataset[FEATURES].astype(float)
            y = dataset["label"].astype(str)
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.25, random_state=42, stratify=y
            )
            classifier = DecisionTreeClassifier(
                max_depth=5, min_samples_leaf=3, class_weight="balanced", random_state=42
            )
            classifier.fit(x_train, y_train)
            y_pred = classifier.predict(x_test)
            st.session_state.model = classifier
            st.session_state.model_metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
                "train_count": len(x_train),
                "test_count": len(x_test),
            }
            joblib.dump(classifier, MODEL_FILE)
            st.success("Model trained and saved. Evaluate it on varied, unseen room conditions.")

    if st.session_state.model_metrics:
        metrics = st.session_state.model_metrics
        a, b, c = st.columns(3)
        a.metric("Hold-out accuracy", f'{metrics["accuracy"]:.1%}')
        b.metric("Training samples", metrics["train_count"])
        c.metric("Test samples", metrics["test_count"])
        st.dataframe(pd.DataFrame(metrics["report"]).transpose(), use_container_width=True)
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_tree(st.session_state.model, feature_names=FEATURES, class_names=st.session_state.model.classes_, filled=True, rounded=True, ax=ax)
        st.pyplot(fig)
        with open(MODEL_FILE, "rb") as file:
            st.download_button("Download trained model", file.read(), file_name=MODEL_FILE.name)
    else:
        st.info("Collect labeled examples, then train the model.")

with info_tab:
    st.subheader("Project purpose and sensor limits")
    st.markdown("""
- **PIR:** detects movement, not continuous human presence.
- **HC-SR04:** measures distance to the nearest reflecting object; it does not count people.
- **LDR:** provides a relative light-level reading.
- **DHT11:** temperature and humidity are displayed if the firmware outputs them.
- **Decision Tree:** learns from labeled examples to predict EMPTY or OCCUPIED.
- **This version:** machine learning runs on the PC. ESP32 deployment and real appliance switching are separate later steps.
- Do not connect mains-powered appliances directly to ESP32 pins.
""")
    st.code("time_ms,motion,distance_cm,light_adc", language="text")
    st.caption("Also accepts the longer CSV output from the earlier full firmware.")

st.caption("Dashboard refreshes automatically. For reliable ML, label real room states and collect varied examples.")
