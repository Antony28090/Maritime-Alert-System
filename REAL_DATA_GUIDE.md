# How to Use Real-Life GPS Data

Currently, the system uses a **Simulation** to generate boat movement. To use **Real-Life Data** (e.g., from a GPS module on a boat), follow these steps.

## 1. Hardware Requirements
You will need a GPS Receiver module. Common options include:
-   **Neo-6M GPS Module**: Cheap and works well with Raspberry Pi/Arduino.
-   **USB GPS Receiver**: Plugs directly into a laptop.

## 2. Install Python GPS Library
If you are using a Raspberry Pi/Laptop with a GPS module connected via Serial (USB/UART):
```bash
pip install pyserial pynmea2
```

## 3. Modify the Code (`app.py`)

You need to edit the `SimulationThread` class in `app.py`. Instead of generating a random point, you will read from the GPS.

### Step A: Add a GPS Reading Function
Add this function to `app.py` (or a sidebar utility):

```python
import serial
import pynmea2

# Configure your Serial Port (Check device manager/ls /dev/tty*)
# Windows: 'COM3', Linux/RPi: '/dev/ttyUSB0' or '/dev/ttyS0'
def get_live_gps_coordinates():
    try:
        with serial.Serial('COM3', baudrate=9600, timeout=1) as ser:
            for _ in range(10): # Try reading a few lines
                line = ser.readline().decode('utf-8')
                if line.startswith('$GPGGA'): # NMEA format for Position
                    msg = pynmea2.parse(line)
                    if msg.latitude and msg.longitude:
                        return msg.latitude, msg.longitude
    except Exception as e:
        print(f"GPS Error: {e}")
    return None, None
```

### Step B: Update the Loop
In `app.py`, find the `SimulationThread` class. Change the `run` method:

**Current (Simulation):**
```python
# ... inside the loop ...
# Simulate next point based on previous
next_lat = self.current_lat + (velocity_lat * 0.01)
next_lon = self.current_lon + (velocity_lon * 0.01)
```

**New (Real Data):**
```python
# ... inside the loop ...
real_lat, real_lon = get_live_gps_coordinates()

if real_lat is not None:
    self.path.append((real_lat, real_lon))
    # Update current position for the Zone Classifier
    # ... rest of the logic remains the same ...
else:
    print("Waiting for GPS signal...")
```

## 4. Testing
1.  Connect your GPS hardware.
2.  Run `app.py`.
3.  The map should now show your physical location moving in real-time!
