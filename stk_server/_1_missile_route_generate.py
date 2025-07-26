import json
import random

from stk_server.Packages.Tools import haversine_distance
from utils.misc_utils import get_current_timestamp, get_data_dir

# Define launch bases (latitude, longitude) - Massively expanded for total global domination
launch_bases = [
    (31.8018, -106.3956),  # Fort Bliss, USA
    (64.6633, -147.1015),  # Eielson AFB, USA
    (49.4369, 7.6003),  # Ramstein Air Base, Germany
    (35.2837, 139.6672),  # Yokosuka Naval Base, Japan
    (-7.3196, 72.4229),  # Diego Garcia, British Indian Ocean Territory
    (76.5318, -68.7032),  # Thule Air Base, Greenland
    (25.1174, 51.3149),  # Al Udeid Air Base, Qatar
    (34.5904, 32.9954),  # RAF Akrotiri, Cyprus
    (21.3499, -157.9606),  # Pearl Harbor, USA
    (40.7029, 141.3684),  # Misawa Air Base, Japan
    (37.0003, 35.4259),  # Incirlik Air Base, Turkey
    (36.9607, 126.9169),  # Camp Humphreys, South Korea
    (34.1448, 132.2200),  # MCAS Iwakuni, Japan
    (36.2360, -115.0343),  # Nellis AFB, USA
    (35.1391, -79.0100),  # Fort Liberty (formerly Bragg), USA
    (36.9465, -76.2893),  # Naval Station Norfolk, USA
    (27.9275, -15.3866),  # Gando Air Base, Spain (Canary Islands)
    (14.3827, -87.6212),  # Soto Cano Air Base, Honduras
    (47.1078, -122.5769),  # Joint Base Lewis-McChord, USA
    (45.8225, 47.1150),  # Akhtubinsk (Russian missile test site), Russia
    (31.3260, 119.8202),  # Lop Nur (Chinese nuclear/missile site), China
    (35.6895, 51.3890),  # Tehran Missile Base (approximate), Iran
    (39.0917, 125.7642),  # Pyongyang Missile Complex (approximate), North Korea
    (54.8333, 20.5000),  # Kaliningrad Naval Base, Russia
    (22.5431, 113.0806),  # Yulin Naval Base (submarine/missile site), China
    (24.4539, 54.3773),  # Abu Dhabi (UAE military hub), UAE
    (50.8439, 4.4213),  # Kleine Brogel Air Base (NATO nuclear site), Belgium
    (35.4955, 33.2731),  # Larnaca (potential allied base), Cyprus
    (68.6167, 27.4167),  # Ivalo Air Base, Finland (NATO-aligned)
    (21.3169, 40.5508),  # Taif Air Base, Saudi Arabia
    (34.7903, 126.3812),  # Kunsan Air Base, South Korea
    (51.1690, 6.0989),  # Geilenkirchen Air Base (NATO AWACS), Germany
    (59.9139, 30.3158),  # St. Petersburg Missile Site (approximate), Russia
    (40.6413, -73.7781),  # JFK Area (hypothetical U.S. East Coast launch), USA
    # New additions: Even more launch sites for unrelenting barrages
    (44.4071, 42.8746),  # Mozdok Air Base, Russia
    (28.4667, 77.0333),  # Hindon Air Force Station, India
    (24.8608, 67.2106),  # Masroor Air Base, Pakistan
    (33.5138, 36.2783),  # Damascus Military Site (approximate), Syria
    (37.5052, 126.6249),  # Osan Air Base, South Korea
    (26.7122, 50.1403),  # King Abdulaziz Air Base, Saudi Arabia
    (48.3538, 11.7861),  # Neubiberg Air Base (historical NATO), Germany
    (35.8339, 14.5347),  # Luqa Air Base, Malta (NATO support)
    (70.4966, 25.6133),  # Kirkenes Air Base, Norway (Arctic NATO)
    (32.3008, 34.8880),  # Palmachim Air Base, Israel
    (55.4103, -3.3430),  # Faslane Naval Base (UK nuclear subs), UK
    (38.8740, -104.4105),  # Cheyenne Mountain Complex, USA
    (27.8493, -80.4357),  # Cape Canaveral (launch site), USA
    (46.7434, 142.7222),  # Yuzhno-Sakhalinsk Base, Russia
    (19.0974, 72.8656),  # Mumbai Naval Base, India
    (35.2338, 129.0825),  # Busan Naval Base, South Korea
    (21.4526, -158.0393),  # Schofield Barracks, USA (Hawaii)
    (51.8836, -176.6364),  # Eareckson Air Station (Aleutians), USA
]

# Define target cities (latitude, longitude) - Bloated with more targets, including extra Chinese strategic sites
target_cities = [
    (38.9072, -77.0369),  # Washington DC, USA
    (39.9042, 116.4074),  # Beijing, China
    (55.7558, 37.6173),  # Moscow, Russia
    (28.6139, 77.2090),  # New Delhi, India
    (32.0853, 34.7818),  # Tel Aviv, Israel
    (51.5074, -0.1278),  # London, UK
    (35.6895, 139.6917),  # Tokyo, Japan
    (48.8566, 2.3522),  # Paris, France
    (52.5200, 13.4050),  # Berlin, Germany
    (37.5665, 126.9780),  # Seoul, South Korea
    (-35.2809, 149.1300),  # Canberra, Australia
    (24.7136, 46.6753),  # Riyadh, Saudi Arabia
    (8.9806, 38.7578),  # Addis Ababa, Ethiopia
    (-34.6037, -58.3816),  # Buenos Aires, Argentina
    (-15.7939, -47.8828),  # Brasilia, Brazil
    (45.4215, -75.6972),  # Ottawa, Canada
    (19.4326, -99.1332),  # Mexico City, Mexico
    (-25.7479, 28.2293),  # Pretoria, South Africa
    (30.0444, 31.2357),  # Cairo, Egypt
    (41.0082, 28.9784),  # Istanbul, Turkey
    (24.8607, 67.0011),  # Karachi, Pakistan
    (23.8103, 90.4125),  # Dhaka, Bangladesh
    (-6.2088, 106.8456),  # Jakarta, Indonesia
    (21.0278, 105.8342),  # Hanoi, Vietnam
    (41.9028, 12.4964),  # Rome, Italy
    (40.4168, -3.7038),  # Madrid, Spain
    (37.9838, 23.7275),  # Athens, Greece
    (6.5244, 3.3792),  # Lagos, Nigeria
    (-33.4489, -70.6693),  # Santiago, Chile
    (-4.4419, 15.2663),  # Kinshasa, Democratic Republic of the Congo
    (35.9078, 127.7669),  # Pyongyang, North Korea
    (35.6895, 51.3890),  # Tehran, Iran
    (33.3152, 44.3661),  # Baghdad, Iraq
    (59.9139, 10.7522),  # Oslo, Norway
    (59.3293, 18.0686),  # Stockholm, Sweden
    (-33.8688, 151.2093),  # Sydney, Australia
    (1.3521, 103.8198),  # Singapore
    (40.7128, -74.0060),  # New York City, USA
    (34.0522, -118.2437),  # Los Angeles, USA
    (31.2304, 121.4737),  # Shanghai, China
    (59.9343, 30.3351),  # St. Petersburg, Russia
    (22.3964, 114.1095),  # Hong Kong, China
    (25.2048, 55.2708),  # Dubai, UAE
    (13.7563, 100.5018),  # Bangkok, Thailand
    (3.1390, 101.6869),  # Kuala Lumpur, Malaysia
    (14.0583, 108.2772),  # Manila, Philippines (approximate central)
    (-23.5505, -46.6333),  # Sao Paulo, Brazil
    (28.0473, -26.2041),  # Johannesburg, South Africa
    (41.8781, -87.6298),  # Chicago, USA
    (43.6532, -79.3832),  # Toronto, Canada
    (50.8503, 4.3517),  # Brussels, Belgium
    (52.3676, 4.9041),  # Amsterdam, Netherlands
    (55.6761, 12.5683),  # Copenhagen, Denmark
    (60.1699, 24.9384),  # Helsinki, Finland
    (64.1355, -21.8954),  # Reykjavik, Iceland
    (23.1291, 113.2644),  # Guangzhou, China (major port and economic hub)
    (22.5431, 114.0579),  # Shenzhen, China (tech and manufacturing center)
    (30.5928, 104.0668),  # Chengdu, China (inland military and aerospace hub)
    (29.5630, 106.5516),  # Chongqing, China (industrial and population giant)
    (30.5728, 114.2790),  # Wuhan, China (central transport and research node)
    (34.3416, 108.9398),  # Xi'an, China (historical and aerospace base)
    (39.0842, 117.2009),  # Tianjin, China (port and industrial powerhouse)
    (38.9140, 121.6147),  # Dalian, China (naval base and shipbuilding center)
    (31.8639, 117.2808),  # Hefei, China (science and quantum tech hub)
    (26.0745, 119.2965),  # Fuzhou, China (coastal military significance)
    (28.1941, 112.9823),  # Changsha, China (media and economic center)
    (36.0611, 103.8343),  # Lanzhou, China (northwestern military gateway)
    (43.8263, 87.6168),  # Urumqi, China (Xinjiang strategic hub)
    (29.6525, 91.1721),  # Lhasa, China (Tibet administrative and military center)
    (25.0443, 102.7097),  # Kunming, China (southwestern border hub)
    # New additions: More Chinese targets plus global extras for balanced carnage
    (32.0603, 118.7969),  # Nanjing, China (eastern command and historical capital)
    (36.6512, 117.1201),  # Jinan, China (Shandong military region)
    (41.7959, 123.4291),  # Shenyang, China (northeastern industrial base)
    (45.8038, 126.5350),  # Harbin, China (Heilongjiang border city)
    (22.8167, 108.3225),  # Nanning, China (Guangxi ASEAN gateway)
    (30.6719, 104.0757),  # Mianyang, China (nuclear research center)
    (31.2222, 121.4581),  # Pudong (Shanghai district), China (financial hub)
    (24.4798, 118.0894),  # Xiamen, China (coastal trade and Taiwan-facing)
    (28.6770, 115.8572),  # Nanchang, China (Jiangxi aviation hub)
    (34.7466, 113.6253),  # Zhengzhou, China (central logistics node)
    (38.0428, 114.5149),  # Shijiazhuang, China (Hebei military area)
    (40.8112, 111.6522),  # Hohhot, China (Inner Mongolia capital)
    (38.4865, 106.2327),  # Yinchuan, China (Ningxia autonomous region)
    (47.3532, 123.1872),  # Qiqihar, China (northeastern aviation base)
    (39.6133, 109.7809),  # Yulin, China (Shaanxi energy and military site)
    # Global additions for extra chaos
    (19.0760, 72.8777),  # Mumbai, India
    (35.6892, 51.3890),  # Tehran (duplicate for emphasis? No, refined), Iran
    (37.7749, -122.4194),  # San Francisco, USA
    (51.1789, -1.8262),  # London (alternate? Wait, adding Liverpool instead: 53.4084, -2.9916) Liverpool, UK
    (53.4084, -2.9916),  # Liverpool, UK
    (35.2271, -80.8431),  # Charlotte, USA (financial hub)
    (25.7617, -80.1918),  # Miami, USA
    (55.9533, -3.1883),  # Edinburgh, UK
    (53.3498, -6.2603),  # Dublin, Ireland
    (47.6062, -122.3321),  # Seattle, USA
    (39.7392, -104.9903),  # Denver, USA
    (-22.9068, -43.1729),  # Rio de Janeiro, Brazil
    (4.7109, -74.0721),  # Bogota, Colombia
    (9.9281, -84.0907),  # San Jose, Costa Rica
    (18.4655, -66.1057),  # San Juan, Puerto Rico
    (64.1265, -21.8174),  # Reykjavik (duplicate? Adding Akureyri: 65.6885, -18.1262) Akureyri, Iceland
    (65.6885, -18.1262),  # Akureyri, Iceland
]

# Generate 25 unique routes (base-target pairs) for volleys
num_routes = 40
routes = []
used_pairs = set()  # To ensure uniqueness
min_distance_threshold = 1000  # km, filter out routes unsuitable for long-range missiles
while len(routes) < num_routes:
    base = random.choice(launch_bases)
    target = random.choice(target_cities)
    pair = (base, target)
    if pair not in used_pairs:
        # Calculate distance using haversine
        p1 = {"latitude": base[0], "longitude": base[1]}
        p2 = {"latitude": target[0], "longitude": target[1]}
        distance = haversine_distance(p1, p2)
        if distance >= min_distance_threshold:  # Only add if suitable for long-range attack
            used_pairs.add(pair)
            routes.append((base, target, distance))  # Store distance for later use

# Distribute missiles across the routes (random 5-10 per route, no upper limit on total)
missiles_per_route = [random.randint(1, 2) for _ in range(num_routes)]
total_missiles = sum(missiles_per_route)  # This will be >50, typically 125-250

# Now generate the data with sequenced launches
data = []

for route_idx, (base, target, distance) in enumerate(routes):
    current_time = random.randint(0, 10)  # Start time for sequential launches
    missile_id = 1
    num_missiles = missiles_per_route[route_idx]
    for _ in range(num_missiles):
        # Increment time by a stretched random interval (10-30 seconds) for staggered launch
        time_interval = random.randint(100, 150)
        current_time += time_interval
        # Calculate reasonable altitude based on distance (in km, with ±10% variation)
        base_altitude = min(0.12 * distance, 1500.0)  # Cap altitude at 2500 km
        altitude = min(round(base_altitude * random.uniform(0.9, 1.1), 2),
                       1500.0)  # Ensure altitude does not exceed 2500 km
        entry = {
            "name": f"m{route_idx + 1}n{missile_id}",
            "trajectory_epoch_second": current_time,
            "speed": round(random.uniform(3.0, 10.0), 2),  # Random speed
            "altitude": altitude,  # Updated: based on distance
            "latitude": round(base[0] + random.uniform(-0.05, 0.05), 6),  # Slight perturbation
            "longitude": round(base[1] + random.uniform(-0.05, 0.05), 6),
            "impact_latitude": round(target[0] + random.uniform(-0.05, 0.05), 6),
            "impact_longitude": round(target[1] + random.uniform(-0.05, 0.05), 6)
        }
        data.append(entry)
        missile_id += 1

# Optional: Print summary for verification
print(f"Generated {len(data)} missiles across {num_routes} routes.")
for i, num in enumerate(missiles_per_route):
    print(f"Route {i + 1}: {num} missiles")
print(f"Total launch duration: {current_time} seconds (~{current_time // 60} minutes)")

# # Display the JSON array
# print(json.dumps(data, indent=2, ensure_ascii=False))
current_timestamp = get_current_timestamp()
# Create output directory if it doesn't exist
output_dir = get_data_dir() / f"stk_route_data/missile_route_info_v{current_timestamp}.json"

# Save the data to a JSON file
with open(output_dir, "w") as json_file:
    json.dump(data, json_file, indent=2, ensure_ascii=False)

print(f"Data successfully saved to {output_dir}")
