
import random
import json

# Define launch bases (latitude, longitude)
launch_bases = [
    (31.8539, -106.3815),  # Fort Bliss
    (64.6711, -147.0965),  # Eielson AFB
    (49.4478, 7.6009),     # Ramstein
    (35.2812, 139.6700),   # Yokosuka
    (-7.3137, 72.4118),    # Diego Garcia
    (-76.5318, -68.7032),  # Thule
    (25.1202, 51.1846),    # Al Udeid
    (34.6235, 32.9878),    # Akrotiri
    (21.3450, -157.9490),  # Pearl Harbor
    (40.7039, 141.3630),   # Misawa
    (34.9033, 69.4350),    # Bagram
    (36.9931, 127.0920),   # Camp Humphreys
    (34.1832, 132.2083),   # Iwakuni
    (36.2369, -115.0343),  # Nellis
    (35.1400, -79.0000),   # Fort Bragg
    (36.9370, -76.2898),   # Norfolk
    (27.9314, -15.3865),   # Gando
    (14.3392, -87.6494),   # Soto Cano
    (47.1219, -122.5911)   # Lewis-McChord
]

# Define target cities (latitude, longitude)
target_cities = [
    (38.9072, -77.0369),  # Washington DC
    (39.9042, 116.4074),  # Beijing
    (55.7558, 37.6173),   # Moscow
    (28.6139, 77.2090),   # New Delhi
    (32.0853, 34.7818),   # Tel Aviv
    (51.5074, -0.1278),   # London
    (35.6895, 139.6917),  # Tokyo
    (48.8566, 2.3522),    # Paris
    (52.5200, 13.4050),   # Berlin
    (37.5665, 126.9780),  # Seoul
    (-35.2809, 149.1300), # Canberra
    (24.7136, 46.6753),   # Riyadh
    (8.9806, 38.7578),    # Addis Ababa
    (-34.6037, -58.3816), # Buenos Aires
    (-15.7939, -47.8828), # Brasilia
    (45.4215, -75.6972),  # Ottawa
    (19.4326, -99.1332),  # Mexico City
    (-25.7479, 28.2293),  # Pretoria
    (30.0444, 31.2357),   # Cairo
    (41.0082, 28.9784),   # Istanbul
    (24.8607, 67.0011),   # Karachi
    (23.8103, 90.4125),   # Dhaka
    (-6.2088, 106.8456),  # Jakarta
    (21.0278, 105.8342),  # Hanoi
    (41.9028, 12.4964),   # Rome
    (40.4168, -3.7038),   # Madrid
    (37.9838, 23.7275),   # Athens
    (6.5244, 3.3792),     # Lagos
    (-33.4489, -70.6693), # Santiago
    (-4.4419, 15.2663)    # Kinshasa
]

data = []
for i in range(1, 51):
    base = random.choice(launch_bases)
    target = random.choice(target_cities)
    entry = {
        "name": f"m{i}",
        "trajectory_epoch_second": random.randint(0, 300),
        "speed": round(random.uniform(3.0, 10.0), 2),
        "altitude": round(random.uniform(100.0, 1500.0), 2),
        "latitude": round(base[0] + random.uniform(-0.05, 0.05), 6),
        "longitude": round(base[1] + random.uniform(-0.05, 0.05), 6),
        "impact_latitude": round(target[0] + random.uniform(-0.05, 0.05), 6),
        "impact_longitude": round(target[1] + random.uniform(-0.05, 0.05), 6)
    }
    data.append(entry)

# Display the JSON array
print(json.dumps(data, indent=2, ensure_ascii=False))