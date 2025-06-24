import os
import sys
import h5py
import paramiko
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from datetime import datetime
from glob import glob

# === Step 1: Hardcoded Configuration ===
sftp_host = 'mosdac.gov.in'
sftp_port = 22
sftp_user = 'shivali_v'     # <-- Replace with your actual SFTP username
sftp_pass = 'nrsc2169'     # <-- Replace with your actual SFTP password
input_dir = 'C:\\Users\patti\OneDrive\Desktop\ISRO\MOSDAC'   # Local directory to store downloads
remote_folder = "StandingOrder"

# === Step 2: User Input for Date Mode ===
print("Select Date Mode:")
print("1 - Use 3 most recent files")
print("2 - Enter a specific date")
date_mode = input("Enter your choice (1 or 2): ").strip()
use_recent_mode = (date_mode == '1')

if use_recent_mode:
    date = datetime.today()
    date_input = ""
else:
    date_input = input("Enter the date (DD-MM-YYYY): ")
    try:
        date = datetime.strptime(date_input, '%d-%m-%Y')
        target_str = date.strftime('%d%b%Y').upper()
    except ValueError:
        raise ValueError("Invalid date format! Please use DD-MM-YYYY.")

# === Step 3: SFTP Download ===
local_download_path = os.path.join(input_dir, remote_folder)
os.makedirs(local_download_path, exist_ok=True)

try:
    transport = paramiko.Transport((sftp_host, sftp_port))
    transport.connect(username=sftp_user, password=sftp_pass)
    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.chdir(remote_folder)

    files = sftp.listdir()
    if use_recent_mode:
        download_files = [f for f in files if f.startswith("3RSND_") and f.endswith(".h5")]
    else:
        download_files = [f for f in files if f.startswith("3RSND_") and target_str in f]

    if not download_files:
        raise FileNotFoundError("No matching files found on the server.")

    for filename in download_files:
        local_path = os.path.join(local_download_path, filename)
        if not os.path.exists(local_path):
            print(f"⬇️ Downloading {filename}")
            sftp.get(filename, local_path)
        else:
            print(f"⏭️ Skipping {filename} (already exists)")

    print("✅ All relevant files downloaded.")
except Exception as e:
    print("❌ SFTP Error:", e)
    # sys.exit()
finally:
    try:
        sftp.close()
        transport.close()
    except:
        pass

# === Step 4: Local File Selection ===
def extract_datetime_from_filename(filename):
    try:
        parts = filename.split('_')
        date_str = parts[1]
        time_str = parts[2]
        return datetime.strptime(date_str + time_str, "%d%b%Y%H%M")
    except:
        return datetime.min

if use_recent_mode:
    all_files = glob(os.path.join(local_download_path, "3RSND_*_L2B_SA1_V01R00.h5"))
    sorted_files = sorted(all_files, key=lambda x: extract_datetime_from_filename(os.path.basename(x)), reverse=True)
    selected_files = sorted_files[:3]
else:
    pattern = f"3RSND_{target_str}_*_L2B_SA1_V01R00.h5"
    selected_files = glob(os.path.join(local_download_path, pattern))

if not selected_files:
    print("❌ No matching files found locally after filtering. Please check the date or file availability.")
    sys.exit()

# === Step 5: TPW Data Processing ===
def get_pw_data(file, dataset_name):
    if dataset_name in file:
        data = np.array(file[dataset_name][0, :, :], dtype=np.float32)
        data[data == -999.0] = np.nan
        return data
    return None

def get_lat_lon(file):
    lat = file['Latitude'][:, :] / 100.0
    lon = file['Longitude'][:, :] / 100.0
    return lat, lon

pw_accumulated = None
for path in selected_files:
    print(f"📖 Reading: {os.path.basename(path)}")
    with h5py.File(path, 'r') as f:
        pw1 = get_pw_data(f, 'L1_PREC_WATER')
        pw2 = get_pw_data(f, 'L2_PREC_WATER')
        pw3 = get_pw_data(f, 'L3_PREC_WATER')
        pw_stack = np.stack([pw1, pw2, pw3])
        pw_total = np.nansum(pw_stack, axis=0)
        pw_accumulated = pw_total if pw_accumulated is None else pw_accumulated + pw_total

pw_average = pw_accumulated / len(selected_files)

with h5py.File(selected_files[0], 'r') as ref_file:
    lat_grid, lon_grid = get_lat_lon(ref_file)

# === Step 6: Plotting ===
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 5))
cf = ax.contourf(lon_grid, lat_grid, pw_average, levels=np.arange(0, 70), cmap='jet')
ax.coastlines()

gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.left_labels = True
gl.bottom_labels = True
gl.xlocator = mticker.MultipleLocator(5)
gl.ylocator = mticker.MultipleLocator(5)

today_str = datetime.today().strftime("%Y-%m-%d")
title_date = date_input if not use_recent_mode else f"Most Recent 3 hours for {today_str}"
plt.colorbar(cf, label='Total Precipitable Water (mm)')
plt.title(f"TPW Average – {title_date}")
plt.tight_layout()
plt.show()
