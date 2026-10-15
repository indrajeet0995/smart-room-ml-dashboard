SMART ROOM ML DASHBOARD - FINAL PACKAGE

INSTALL / RUN (Windows)
1. Extract this ZIP into a new folder.
2. Open PowerShell in the folder containing app.py.
3. Run:
   python -m pip install -r requirements.txt
4. Close Arduino IDE Serial Monitor / Serial Plotter.
5. Run:
   python -m streamlit run app.py
6. Open http://localhost:8501 if it does not open automatically.
7. Choose the ESP32 COM port in the dashboard sidebar and click Connect to ESP32.

Supported ESP32 serial formats:
  time_ms,motion,distance_cm,light_adc
  time_ms,motion,distance_cm,light_adc,temperature,humidity,dark,status

DATA AND ML
- Label samples using the real room state, not the model prediction.
- Collect both EMPTY and OCCUPIED examples under varied conditions; aim for 100+ per class.
- Model is trained and inferred on the PC in this package.
- This version does not yet send predictions back to ESP32 or control mains appliances.
- HC-SR04 measures nearest object distance; it is not a reliable people counter.

TROUBLESHOOTING
- If requirements.txt is not found, PowerShell is not in the extracted folder. Run `dir` and check that app.py and requirements.txt are listed.
- If COM port is busy, close Arduino IDE Serial Monitor and Serial Plotter.
- If no COM port appears, check USB data cable and ESP32 USB serial driver.
