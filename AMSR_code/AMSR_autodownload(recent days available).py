import os
import requests
from datetime import datetime, timedelta
from tqdm import tqdm

# ====== Settings ======
output_dir = "C:\\Users\\patti\\OneDrive\\Desktop\\ISRO\\AMSR_TPW"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(output_dir):
    file_path = os.path.join(output_dir, filename)
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")

CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
DATASET_CONCEPT_ID = "C3243533753-NSIDC_CPRD"  # AMSR Unified L2B Ocean (TPW included)

# ====== Earthdata credentials ======
def get_auth():
    #  Replace with your actual Earthdata login credentials
    username = "advaitha"
    password = "Advaitha@$NITW2023"
    return (username, password)

# ====== CMR Search Function ======
def cmr_search(temporal, page_num=1):
    params = {
        "collection_concept_id": DATASET_CONCEPT_ID,
        "temporal": temporal,
        "page_size": 2000,
        "page_num": page_num
    }
    resp = requests.get(CMR_URL, params=params)
    resp.raise_for_status()
    return resp.json()

# ====== Get File URLs ======
def collect_urls(date_str):
    temporal = f"{date_str}T00:00:00Z,{date_str}T23:59:59Z"
    urls = []
    page = 1
    while True:
        resp = cmr_search(temporal, page)
        granules = resp["feed"]["entry"]
        if not granules:
            break
        for g in granules:
            for loc in g["links"]:
                if loc.get("href", "").endswith(".he5"):
                    urls.append(loc["href"])
        page += 1
    return urls

# ====== File Downloader ======
def download_file(url):
    local_name = os.path.join(output_dir, url.split("/")[-1])
    if os.path.exists(local_name):
        print(f"Already downloaded: {local_name}")
        return
    with requests.get(url, auth=get_auth(), stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(local_name, "wb") as f, tqdm(
            desc=local_name, total=total, unit='iB', unit_scale=True
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)
                bar.update(len(chunk))

# Ask user for reference date input
reference_date_str = input("Enter the reference date (YYYY-MM-DD): ")

# Validate and parse the date
try:
    refer_date = datetime.strptime(reference_date_str, "%Y-%m-%d").date()
except ValueError:
    print("Invalid date format. Please enter date in YYYY-MM-DD format.")
    exit()

# ====== Search Backward Until Data Found ======
days_back = 0
max_days = 30  # Safety cap, don't go back more than 30 days

while days_back < max_days:
    #reference_date_str="2024-07-30"
    refer_date=datetime.strptime(reference_date_str,"%Y-%m-%d").date()
    
    date =(refer_date -timedelta(days=days_back)).strftime("%Y-%m-%d")
    print(f"\n Checking for granules on {date}...")
    urls = collect_urls(date)

    if urls:
        print(f" Found {len(urls)} granules on {date}. Downloading...")
        for u in urls:
            download_file(u)
        break  # Stop after first date with data
    else:
        print(f" No granules found on {date}. Trying previous day...")
        days_back += 1

if days_back >= max_days:
    print("No data found in the last 30 days.")
    

#--------------------------------------------------------------------
#----------------------AMSR plotting---------------------------------
#---------------------------------------------------------------------


import h5py
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from datetime import datetime


# === Data directory ===
# datadir = r"C:\Users\patti\OneDrive\Desktop\ISRO\AMSR_TPW"
# os.chdir(datadir)

# === File selection ===
selected_files = sorted(f for f in os.listdir(output_dir) if f.endswith('.he5'))

print(f"Found {len(selected_files)} HE5 files.")

# === Dataset paths ===
tpw_path = '/HDFEOS/SWATHS/AMSR2_Level2_Ocean_Suite/Data Fields/TotalPrecipitableWater'
lat_path = '/HDFEOS/SWATHS/AMSR2_Level2_Ocean_Suite/Geolocation Fields/Latitude'
lon_path = '/HDFEOS/SWATHS/AMSR2_Level2_Ocean_Suite/Geolocation Fields/Longitude'

# === Plot limits ===
plot_lon_min, plot_lon_max = 50, 100
plot_lat_min, plot_lat_max = -15, 40

# Parse date from the first file that matches the naming convention
if selected_files:
    first_file = selected_files[-1]
    #print(first_file)
    # Example filename: AMSR_U2_L2_Ocean_V01_202001120435_A.he5
    try:
        datetime_str = first_file.split("_")[5]
        dt_obj = datetime.strptime(datetime_str, "%Y%m%d%H%M")
        plot_date = dt_obj.strftime("%d %b %Y")
    except (IndexError, ValueError):
        plot_date = "Unknown Date"
else:
    plot_date = "No Data"


# === Create plot ===
fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')  # High-res coastlines only
ax.set_extent([plot_lon_min, plot_lon_max, plot_lat_min, plot_lat_max], crs=ccrs.PlateCarree())

# === Loop through files and overlay TPW ===
for file in selected_files:
    file_path = os.path.join(output_dir, file)
    with h5py.File(file_path, 'r') as f:
        tpw = np.array(f[tpw_path][:], dtype=np.float32)
        lat = np.array(f[lat_path][:], dtype=np.float32)
        lon = np.array(f[lon_path][:], dtype=np.float32)
        
        # Mask invalid data
        tpw[tpw < 0] = np.nan

        # Apply spatial bounds
        mask = (lon >= plot_lon_min) & (lon <= plot_lon_max) & \
               (lat >= plot_lat_min) & (lat <= plot_lat_max)
        
        tpw_masked = np.where(mask, tpw, np.nan)

        # Overlay TPW data
        cf = ax.pcolormesh(lon, lat, tpw_masked, cmap='jet', shading='auto',
                           alpha=0.5, transform=ccrs.PlateCarree())

# === Add colorbar and labels ===
plt.colorbar(cf, ax=ax, label='Total Precipitable Water (mm)')

# Set axis ticks
# Axis labels and title
ax.set_xlabel("Longitude (E)")
ax.set_ylabel("Latitude (N)")
ax.set_title(f"TPW on {plot_date}", fontweight='bold')

# === Save the figure ===
save_dir = r"C:\Users\patti\OneDrive\Desktop\ISRO\AMSR_TPW\Plots"
os.makedirs(save_dir, exist_ok=True)
output_filename = f"TPW_{plot_date.replace(' ', '_')}.png"
output_path = os.path.join(save_dir, output_filename)
plt.savefig(output_path, dpi=300, bbox_inches='tight')

# === Show plot ===
plt.show()

