// Initialize Map
var map = L.map('map').setView([9.2872, 79.3130], 9); // Start near Rameswaram

// Light Mode Tiles (OpenStreetMap)
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);

// Icons
var boatIcon = L.icon({
    iconUrl: 'https://cdn-icons-png.flaticon.com/512/2990/2990438.png',
    iconSize: [32, 32],
    iconAnchor: [16, 16]
});

// Markers & Layers
var vesselMarker = L.marker([0, 0], { icon: boatIcon }).addTo(map);
var imblLine = null;
var dangerZoneLayer = null;
var pathLine = L.polyline([], { color: '#00bcd4', weight: 4 }).addTo(map); // Cyan
var forecastLine = L.polyline([], { color: '#ef6c00', dashArray: '5, 10', weight: 4 }).addTo(map); // Dark Orange

// Voice Alert State
var lastSpeechTime = 0;
var speechCooldown = 15000; // 15 seconds
var lastAlertState = null;

function speakAlert(message, force) {
    if (!window.speechSynthesis) return;

    // Only speak if forced (i.e. state has changed)
    if (!force) return;

    var now = Date.now();

    // Cancel any current speech
    window.speechSynthesis.cancel();

    var utterance = new SpeechSynthesisUtterance(message);
    utterance.lang = 'ta-IN'; // Tamil India
    utterance.rate = 1.0;

    window.speechSynthesis.speak(utterance);
    lastSpeechTime = now;
}

// Fetch Configuration (Boundaries)
fetch('/api/config')
    .then(response => response.json())
    .then(config => {
        if (config.imbl_points) {
            // Visualize Danger Zone (Red transparent strip) - "Zone Visualization"
            dangerZoneLayer = L.polyline(config.imbl_points, {
                color: 'red',
                weight: 60, // Wide strip
                opacity: 0.2,
                lineCap: 'butt'
            }).addTo(map);
            dangerZoneLayer.bindPopup("Danger Zone (< 2km)");

            // Draw IMBL
            imblLine = L.polyline(config.imbl_points, {
                color: 'red',
                weight: 3,
                dashArray: '10, 10'
            }).addTo(map);
            imblLine.bindPopup("International Maritime Boundary Line (India-Sri Lanka)");

            // Fit bounds
            map.fitBounds(imblLine.getBounds());
        }
    });

// Poll Status
function updateStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            var lat = data.lat;
            var lon = data.lon;

            // Clear path if this is a new trip
            var iconElement = vesselMarker.getElement();
            if (data.step === 0) {
                pathLine.setLatLngs([]);
                // Disable transition to instantly snap boat to the new start point
                if (iconElement) iconElement.style.transition = 'none';
            } else {
                // Enable smooth 1-second CSS transitions for ~60fps movement interpolation
                if (iconElement) iconElement.style.transition = 'transform 1s linear';
            }

            // Update Marker
            var newLatLng = new L.LatLng(lat, lon);
            vesselMarker.setLatLng(newLatLng);
            pathLine.addLatLng(newLatLng);

            // Update Prediction Line
            if (data.prediction && data.prediction.length > 0) {
                var predPoints = [newLatLng];
                data.prediction.forEach(p => predPoints.push(new L.LatLng(p[0], p[1])));
                forecastLine.setLatLngs(predPoints);
            }

            // Update HUD
            document.getElementById('loc-val').innerText = `${lat.toFixed(4)}, ${lon.toFixed(4)}`;

            var zoneEl = document.getElementById('zone-val');
            zoneEl.innerText = data.zone;
            zoneEl.className = 'value zone-' + data.zone.toLowerCase();

            document.getElementById('forecast-val').innerText = data.forecast_msg;

            // Alert Box
            var alertBox = document.getElementById('alert-box');
            var currentAlertLevel = data.alert_level;

            if (currentAlertLevel === 'crossed') {
                alertBox.style.display = 'block';
                alertBox.className = 'alert-danger';
                alertBox.innerText = "BOUNDARY CROSSED! TURN BACK!";

                var forceAudio = (lastAlertState !== 'crossed');
                speakAlert("எச்சரிக்கை! நீங்கள் எல்லையைத் தாண்டிவிட்டீர்கள். உடனடியாகத் திரும்பிச் செல்லவும்.", forceAudio);

            } else if (currentAlertLevel === 'danger') {
                alertBox.style.display = 'block';
                alertBox.className = 'alert-danger';
                alertBox.innerText = "DANGER! HIGH RISK OF CROSSING!";

                var forceAudio = (lastAlertState !== 'danger');
                speakAlert("எச்சரிக்கை! நீங்கள் எல்லை தாண்டும் அபாயத்தில் உள்ளீர்கள். உடனே திரும்புங்கள்.", forceAudio);

            } else if (currentAlertLevel === 'caution') {
                alertBox.style.display = 'block';
                alertBox.className = 'alert-caution';
                alertBox.innerText = "CAUTION: APPROACHING BOUNDARY";

                // Optional: Caution voice
                // speakAlert("கவனிக்கவும். நீங்கள் எல்லையை நெருங்குகிறீர்கள்.");

            } else {
                alertBox.style.display = 'none';
            }

            lastAlertState = currentAlertLevel;
        })
        .catch(err => console.error("Error fetching status:", err));
}

// Update every 1 second
setInterval(updateStatus, 1000);
