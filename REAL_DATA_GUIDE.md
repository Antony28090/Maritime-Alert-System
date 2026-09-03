# How to Use Real-Life GPS Data

By default the system runs `SimulationThread` in `app.py`, which calls `generate_trajectory()` from `src/data_generator.py` to produce a synthetic path. To feed **real GPS data** from a hardware receiver instead, replace the trajectory source inside the thread's `run()` loop.

## 1. Hardware Requirements
Any receiver that emits NMEA sentences over serial works. Common options:
- **Neo-6M / Neo-M8N GPS module** — pairs well with Raspberry Pi or Arduino.
- **USB GPS receiver** — plugs directly into a laptop and appears as a serial COM port.

## 2. Install the Extra Python Libraries
```bash
pip install pyserial pynmea2
```

## 3. Add a GPS Reader

Add a helper near the top of `app.py`:

```python
import serial
import pynmea2

# Windows: 'COM3'.  Linux/Raspberry Pi: '/dev/ttyUSB0' or '/dev/ttyS0'.
GPS_PORT = 'COM3'

def read_gps_fix():
    try:
        with serial.Serial(GPS_PORT, baudrate=9600, timeout=1) as ser:
            for _ in range(10):
                line = ser.readline().decode('utf-8', errors='ignore')
                if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                    msg = pynmea2.parse(line)
                    if msg.latitude and msg.longitude:
                        return msg.latitude, msg.longitude
    except Exception as e:
        print(f"GPS Error: {e}")
    return None, None
```

## 4. Swap the Trajectory Source

In `SimulationThread.run()` (see `app.py`), the current loop iterates a synthetic list:

```python
trajectory_data = generate_trajectory(trip_id=trip_id, n_points=300, force_crossing=is_forced)
for i, step in enumerate(trajectory_data):
    lat = step['lat']
    lon = step['lon']
    # ... zone classification, forecasting, alerts ...
```

Replace the outer generator + `for` with a live-polling loop:

```python
i = 0
while self.running:
    lat, lon = read_gps_fix()
    if lat is None:
        time.sleep(1)
        continue
    # ... reuse the rest of the loop body (zone classification, forecasting, alerts) ...
    i += 1
```

Everything downstream — `zone_model.predict(lat, lon)`, the LSTM path buffer, the alert triggers, the `simulation_state` update — stays the same.

## 5. Test
1. Connect the GPS receiver and confirm the serial port name.
2. Run `python app.py`.
3. Open `http://127.0.0.1:5000` — the boat marker should track your physical location.
