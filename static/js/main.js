// Initialize Map
var map = L.map('map').setView([9.2872, 79.3130], 10); // Start near Rameswaram

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

// Fetch Configuration (Boundaries)
fetch('/api/config')
    .then(response => response.json())
    .then(config => {
        // Visualize Danger Zone (Red transparent strip) - "Zone Visualization"
        dangerZoneLayer = L.polyline([
            config.imbl_start,
            config.imbl_end
        ], {
            color: 'red',
            weight: 80, // Wide strip
            opacity: 0.2,
            lineCap: 'butt'
        }).addTo(map);
        dangerZoneLayer.bindPopup("Danger Zone (< 2km)");

        // Draw IMBL
        imblLine = L.polyline([
            config.imbl_start,
            config.imbl_end
        ], { color: 'red', weight: 3, dashArray: '10, 10' }).addTo(map);
        imblLine.bindPopup("IMBL (Maritime Boundary)");

        // Fit bounds
        map.fitBounds(imblLine.getBounds());
    });

// Poll Status
function updateStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            var lat = data.lat;
            var lon = data.lon;

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
            if (data.alert_level === 'danger') {
                alertBox.style.display = 'block';
                alertBox.className = 'alert-danger';
                alertBox.innerText = "DANGER! TURN BACK!";
            } else if (data.alert_level === 'caution') {
                alertBox.style.display = 'block';
                alertBox.className = 'alert-caution';
                alertBox.innerText = "CAUTION: APPROACHING BOUNDARY";
            } else {
                alertBox.style.display = 'none';
            }
        })
        .catch(err => console.error("Error fetching status:", err));
}

// Update every 1 second
setInterval(updateStatus, 1000);
