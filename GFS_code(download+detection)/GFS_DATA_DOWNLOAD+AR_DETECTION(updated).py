import requests
import os
from datetime import datetime, timedelta

# Set output directory
datadir = r'C:\Users\patti\OneDrive\Desktop\ISRO\GFSdata(new)'
os.makedirs(datadir, exist_ok=True)

# Clear existing files in the directory
for filename in os.listdir(datadir):
    file_path = os.path.join(datadir, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)

os.chdir(datadir)

# Get today's, yesterday's, and tomorrow's dates
today = datetime.today()
yesterday = today - timedelta(days=1)
tomorrow = today + timedelta(days=1)

# Format dates
today_str = today.strftime("%Y%m%d")
yesterday_str = yesterday.strftime("%Y%m%d")
tomorrow_str = tomorrow.strftime("%Y%m%d")

def gfs_url_available(date_str, hour=0):
    test_url = (
        f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?"
        f"dir=%2Fgfs.{date_str}%2F00%2Fatmos&file=gfs.t00z.pgrb2.0p25.f{str(hour).zfill(3)}"
        f"&lev_850_mb=on&var_UGRD=on&subregion=&toplat=20&leftlon=60&rightlon=80&bottomlat=10"
    )
    try:
        r = requests.head(test_url)
        return r.status_code == 200
    except:
        return False

def download_gfs_files(base_date, hours, logical_date_str, output_dir):
    for i in hours:
        fhour = str(i).zfill(3)
        logical_fname = f"gfs.{logical_date_str}.t00z.pgrb2.0p25.f{fhour}"
        url = (
            f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?"
            f"dir=%2Fgfs.{base_date}%2F00%2Fatmos&file=gfs.t00z.pgrb2.0p25.f{fhour}"
            f"&var_PRES=on&var_SPFH=on&var_UFLX=on&var_UGRD=on&var_U-GWD=on"
            f"&var_VFLX=on&var_VGRD=on&var_V-GWD=on"
            f"&lev_1000_mb=on&lev_975_mb=on&lev_950_mb=on&lev_925_mb=on"
            f"&lev_900_mb=on&lev_850_mb=on&lev_800_mb=on&lev_750_mb=on"
            f"&lev_700_mb=on&lev_650_mb=on&lev_600_mb=on&lev_550_mb=on&lev_500_mb=on"
            f"&subregion=&toplat=40&leftlon=50&rightlon=100&bottomlat=-15"
        )
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(os.path.join(output_dir, logical_fname), "wb") as f:
                f.write(response.content)
            print(f"Downloaded: {logical_fname}")
        except:
            pass  # No error or fallback print

download_gfs_files(yesterday_str, range(0, 24), yesterday_str, datadir)

# Use today's forecast if available, else use yesterday’s extended forecast
if gfs_url_available(today_str):
    base_date_str = today_str
    start_hour_today = 0
    start_hour_tomorrow = 24
else:
    base_date_str = yesterday_str
    start_hour_today = 24
    start_hour_tomorrow = 48

# Download data for logical today and tomorrow
download_gfs_files(base_date_str, range(start_hour_today, start_hour_today + 24), today_str, datadir)
download_gfs_files(base_date_str, range(start_hour_tomorrow, start_hour_tomorrow + 24), tomorrow_str, datadir)


#-------------------------------------------------------------------------
#------------------------------AR DETECTION ALGORITHM---------------------
#--------------------------------------------------------------------------


import pygrib
import numpy as np
import os
import scipy.io
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from datetime import date
import geopy.distance
#import iris
#import iris.quickplot as qplt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path
from scipy import ndimage
from scipy.io import loadmat
from netCDF4 import Dataset, num2date
from datetime import datetime

# Define required pressure levels
pre = [500, 600, 650, 700, 750, 800, 850, 900, 950, 975, 1000]


land_mask_dir = r'C:\Users\patti\OneDrive\Desktop\gujarat_to_kerala_coast_mask.mat'
land_mask = loadmat(land_mask_dir)
wc=list(land_mask.items())[3]
land_mask=wc[1]
land_mask=np.flipud(land_mask)

# datadir =r'C:\Users\patti\OneDrive\Desktop\ISRO\GFSdata(new)'
# os.chdir(datadir)

output_dir=r'C:\Users\patti\OneDrive\Desktop\ISRO\GFS_IVT_plots'

datadir2=r'C:\Users\patti\OneDrive\Desktop\ISRO\ARs'

# Define dates
#today = datetime.today()
# yesterday = date - timedelta(days=1)
# tomorrow = date + timedelta(days=1)
current_month=date.month

# Format dates as YYYYMMDD
# today_str = today.strftime("%Y%m%d")
# yesterday_str = yesterday.strftime("%Y%m%d")
# tomorrow_str = tomorrow.strftime("%Y%m%d")
ivt_85th_percentile=0
ivt_min=150
min_size=60
lats=[]
lons=[]

res = 0.25
# Ener the input data lat/lon extent
max_lat = 40
min_lat = -15
max_lon = 100
min_lon = 50

# Function to compute IVT daily mean and spatial mean
def compute_daily_ivt_mean(filepaths):
    ivts = []
    ivtsx=[]
    ivtsy=[]
    u850_list = []
    v850_list = []

    for filepath in filepaths:
        try:
            grbs = pygrib.open(filepath)
            sphum_msgs = grbs.select(name='Specific humidity')
            u_msgs = grbs.select(name='U component of wind')
            v_msgs = grbs.select(name='V component of wind')

            levels_q = [g.level for g in sphum_msgs]
            levels_u = [g.level for g in u_msgs]
            levels_v = [g.level for g in v_msgs]

            if not all(l in levels_q for l in pre) or not all(l in levels_u for l in pre) or not all(l in levels_v for l in pre):
                print(f"Skipping {filepath} — missing levels")
                grbs.close()
                continue

            sphum = [grbs.select(name='Specific humidity', level=l)[0].values for l in pre]
            U = [grbs.select(name='U component of wind', level=l)[0].values for l in pre]
            V = [grbs.select(name='V component of wind', level=l)[0].values for l in pre]
            grbs.close()



            if 850 in pre:
                idx_850 = pre.index(850)
                u850 = U[idx_850]
                v850 = V[idx_850]
                u850_list.append(u850)
                v850_list.append(v850)
            else:
                print(f"850 hPa level missing in {filepath}")
            
            ivt_levels = []
            ivt_levelsx=[]
            ivt_levelsy=[]
            for i in range(len(pre) - 1):
                dp = pre[i + 1] - pre[i]
                qavg = (sphum[i] + sphum[i + 1]) / 2
                uavg = (U[i] + U[i + 1]) / 2 
                vavg = (V[i] + V[i + 1]) / 2
                ivt = (1 / 9.8) * np.sqrt((qavg * uavg * dp * 100)**2 + (qavg * vavg * dp * 100)**2)
                ivtx = (1 / 9.8) * (qavg * uavg * dp * 100)
                                               
                ivty = (1 / 9.8) * (qavg * vavg * dp * 100)
                ivt_levels.append(ivt)
                ivt_levelsx.append(ivtx)
                ivt_levelsy.append(ivty)
                
                

            ivt_sum = np.sum(ivt_levels, axis=0)
            ivts.append(ivt_sum)
            ivtx_sum=np.sum(ivt_levelsx,axis=0)
            ivtsx.append(ivtx_sum)
            ivty_sum=np.sum(ivt_levelsy,axis=0)
            ivtsy.append(ivty_sum)
            

        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue

    if ivts:
        ivt_stack = np.stack(ivts, axis=0)
        ivt_daily_avg = np.mean(ivt_stack, axis=0)
        del ivt_stack
        ivt_stack=np.stack(ivtsx,axis=0)
        eastward_ivt=np.mean(ivt_stack,axis=0)
        del ivt_stack
        ivt_stack=np.stack(ivtsy,axis=0)
        northward_ivt=np.mean(ivt_stack,axis=0)
        u850_mean = np.mean(np.stack(u850_list, axis=0), axis=0)
        v850_mean = np.mean(np.stack(v850_list, axis=0), axis=0)

        return ivt_daily_avg,eastward_ivt,northward_ivt,u850_mean,v850_mean
    else:
        return None,None,None,None,None

# Get list of file paths
yesterday_files = [f for f in os.listdir(datadir) if f.startswith(f"gfs.{yesterday_str}.t00z.pgrb2.0p25")]
tomorrow_files  = [f for f in os.listdir(datadir) if f.startswith(f"gfs.{tomorrow_str}.t00z.pgrb2.0p25")]

yesterday_paths = [os.path.join(datadir, f) for f in yesterday_files if os.path.exists(os.path.join(datadir, f))]
tomorrow_paths = [os.path.join(datadir, f) for f in tomorrow_files if os.path.exists(os.path.join(datadir, f))]

# Compute IVT daily means
ivt_yest_field,ivt_yest_east,ivt_yest_north,u_yest,v_yest= compute_daily_ivt_mean(yesterday_paths)
ivt_tomo_field,ivt_tomo_east,ivt_tomo_north,u_tomo,v_tomo= compute_daily_ivt_mean(tomorrow_paths)

# Combine and calculate 85th percentile
ivt_combined = []
if ivt_yest_field is not None:
    ivt_combined.append(ivt_yest_field)
if ivt_tomo_field is not None:
    ivt_combined.append(ivt_tomo_field)

if ivt_combined:
    ivt_stack = np.stack(ivt_combined, axis=0)
    ivt_85th_percentile = np.percentile(ivt_stack.flatten(), 85)
    
final_threshold=max(ivt_85th_percentile,ivt_min)
    
import os

today_files = [f for f in os.listdir(datadir) if f.startswith(f"gfs.{today_str}.t00z.pgrb2.0p25")]
today_paths = [os.path.join(datadir, f) for f in today_files if os.path.exists(os.path.join(datadir, f))]

ivt_today_field,eastward_ivt,northward_ivt,u_today,v_today=compute_daily_ivt_mean(today_paths)

object_mask=ivt_today_field > final_threshold

ivt_direction_today= ((np.arctan2(eastward_ivt.data, northward_ivt.data) 
                           * 180 + 180/ np.pi)) % 360


import matplotlib.ticker as ticker


# === Plot Today's IVT Field with Only Coastlines ===
if ivt_today_field is not None:
    # Extract lat/lon from one of today's files
    sample_file = today_paths[0]
    grbs_sample = pygrib.open(sample_file)
    lats, lons = grbs_sample.select(name='Specific humidity', level=pre[0])[0].latlons()
    grbs_sample.close()

    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([50, 100, -15, 40], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='10m', linewidth=1.2)

    gl = ax.gridlines(draw_labels=True, linewidth=1.0, color='gray', alpha=0.5, linestyle='-')
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    




    # Create levels from 0 to 900 (inclusive) with 201 steps for smooth gradient
    levels = np.linspace(np.min(ivt_today_field),np.max(ivt_today_field), 200)
    
    # Plot
    cs = ax.contourf(lons, lats, ivt_today_field, levels=levels, cmap='jet', extend='both', transform=ccrs.PlateCarree())
    
    # Add colorbar with evenly spaced integer ticks
    ticks = np.arange(0,np.max(ivt_today_field)+1, 100)
    cbar = plt.colorbar(cs, orientation='vertical', shrink=1.0, pad=0.05, ticks=ticks)
    cbar.set_label('IVT (kg m⁻¹ s⁻¹)', fontsize=20,fontweight='bold')


    #today_str = "20250529"
    date_obj = datetime.strptime(today_str, "%Y%m%d")
    date_text = date_obj.strftime("%Y-%m-%d")
    
    
    plt.title(f"IVT on ({date_text})", fontsize=24, fontweight='bold')
    

    # Save the figure
    save_path = os.path.join(output_dir, f"IVT_Today_{today_str}.png")
    
    ax.text(0.5, -0.1, 'Longitude', transform=ax.transAxes,
        ha='center', va='top', fontsize=20,fontweight='bold')
    
    ax.text(-0.1, 0.5, 'Latitude', transform=ax.transAxes,
        ha='center', va='bottom', rotation=90, fontsize=20,fontweight='bold')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
else:
    print("No valid IVT data found for today.")



from scipy.ndimage import label, sum as nd_sum
import numpy as np

size_filter = []

# Label the connected AR components
labelled_mask, num_objects = label(object_mask)

# Compute the size (number of grid cells) of each object
object_sizes = nd_sum(object_mask, labelled_mask, index=range(1, num_objects + 1))

size_filter = []

for j in range(num_objects):  # j from 0 to num_objects-1, corresponds to object ID j+1
    obj_id = j + 1
    if object_sizes[j] <= min_size:
        size_filter.append(obj_id)
        
landfall_filter = []
landfall_filter = land_mask[0:221,:]*object_mask
       
        
        
#--------------------------Axis Computation---------------------------------------#
#---------------------------------------------------------------------------------#

print('Computing AR Axes')

max_ivt_coords_list = []
axis_list = []
landfall_locations = []
landfall_ivt_magnitudes = []
landfall_ivt_directions = []
axis_coords_list = []

axis_cube = np.zeros_like(land_mask[0:221, :], dtype=int)
landfall_locations_cube = np.zeros_like(land_mask, dtype=int)

step_landfall_ivt_magnitudes = []
step_landfall_ivt_directions = []
step_coords_list = []

def trace_axis(object_ivt, ivt_direction, start_row, start_col, reverse=False):
    traced_axis = np.zeros_like(object_ivt, dtype=int)
    coords = []
    row, col = start_row, start_col
    lat_max, lon_max = object_ivt.shape

    while object_ivt[row, col] > 0:
        if 0 <= row <= lat_max - 1 and 0 <= col <= lon_max - 1:
            object_ivt[row, col] = 0
            traced_axis[row, col] = 1
            coords.append((row, col))

            direction = ivt_direction[row, col]
            if reverse:
                direction = (direction + 180) % 360  # Flip direction

            adjacent_offsets = {
                "N": (-1, 0), "NE": (-1, 1), "E": (0, 1), "SE": (1, 1),
                "S": (1, 0), "SW": (1, -1), "W": (0, -1), "NW": (-1, -1)
            }

            adjacent_cells = {
                key: (row + dx, col + dy) for key, (dx, dy) in adjacent_offsets.items()
                if 0 <= row + dx < lat_max and 0 <= col + dy < lon_max
            }

            directions_map = [
                (337.5, 360, ["NW", "N", "NE"]),
                (0, 22.5, ["NW", "N", "NE"]),
                (22.5, 67.5, ["N", "NE", "E"]),
                (67.5, 112.5, ["NE", "E", "SE"]),
                (112.5, 157.5, ["E", "SE", "S"]),
                (157.5, 202.5, ["SE", "S", "SW"]),
                (202.5, 247.5, ["S", "SW", "W"]),
                (247.5, 292.5, ["SW", "W", "NW"]),
                (292.5, 337.5, ["W", "NW", "N"])
            ]

            next_coords = []
            for lower, upper, neighbors in directions_map:
                if lower < direction <= upper:
                    next_coords = [adjacent_cells[dir] for dir in neighbors if dir in adjacent_cells]
                    break

            if next_coords:
                ivt_vals = [object_ivt[x] for x in next_coords]
                row, col = next_coords[np.argmax(ivt_vals)]
            else:
                break
        else:
            break

    return traced_axis, coords

for j in range(num_objects):
    obj_id = j + 1

    if obj_id in size_filter:
        # Skip small objects for axis and landfall info
        landfall_locations.append((0, 0))
        landfall_ivt_magnitudes.append(0)
        landfall_ivt_directions.append(0)
        step_coords_list.append([])
        axis_coords_list.append([])
        step_landfall_ivt_magnitudes.append(0)
        step_landfall_ivt_directions.append(0)
        max_ivt_coords_list.append((0, 0))
        continue

    # Prepare mask for this object
    object_mask_new = labelled_mask == obj_id
    object_ivt = object_mask_new * ivt_today_field

    # Find max IVT coordinate
    max_value = np.nanmax(object_ivt)
    if max_value > 0:
        max_ivt_coord = np.where(object_ivt == max_value)
        max_ivt_coord = np.column_stack(max_ivt_coord)
        if max_ivt_coord.shape[0] > 0:
            row, col = max_ivt_coord[0][0], max_ivt_coord[0][1]
        else:
            row, col = 0, 0
    else:
        row, col = 0, 0
        max_value = 0

    max_ivt_coords_list.append((row, col))

    # Landfall info
    if land_mask[row, col] == 1:
        landfall_locations.append((row, col))
        landfall_ivt_magnitudes.append(max_value)
        landfall_ivt_directions.append(ivt_direction_today[row, col])
    else:
        landfall_locations.append((0, 0))
        landfall_ivt_magnitudes.append(0)
        landfall_ivt_directions.append(0)

    if (row, col) != (0, 0):
        ivt_copy_forward = object_ivt.copy()
        ivt_copy_backward = object_ivt.copy()

        forward_axis, forward_coords = trace_axis(ivt_copy_forward, ivt_direction_today, row, col, reverse=False)
        backward_axis, backward_coords = trace_axis(ivt_copy_backward, ivt_direction_today, row, col, reverse=True)

        combined_axis = forward_axis + backward_axis
        combined_axis[row, col] = 1

        axis_coords = list(set(forward_coords + backward_coords))
    else:
        combined_axis = np.zeros_like(land_mask[0:221, :], dtype=int)
        axis_coords = []

    new_axis = combined_axis * obj_id
    axis_cube[:221, :] += new_axis[:221, :]

    axis_list.append(new_axis)
    axis_coords_list.append(axis_coords)
    step_landfall_ivt_magnitudes.append(max_value)
    if axis_coords:
        step_landfall_ivt_directions.append(ivt_direction_today[axis_coords[-1][0], axis_coords[-1][1]])
    else:
        step_landfall_ivt_directions.append(0)



#---------------------------AR_Object Length Computation--------------------------#
from skimage.morphology import skeletonize
from geopy.distance import geodesic
import numpy as np

def calculate_ar_lengths(labelled_mask, num_objects, lat, lon, size_filter):
    # Generate 2D lat-lon grid
    
    lat_grid = np.array(lats)  # or lats.copy() if needed
    lon_grid = np.array(lons)


    ar_object_lengths_km = []

    for obj_id in range(1, num_objects + 1):
        if obj_id in size_filter:
            ar_object_lengths_km.append(0)
            continue

        object_mask = (labelled_mask == obj_id)
        if not object_mask.any():
            ar_object_lengths_km.append(0)
            continue

        # Skeletonize the object mask (no slicing!)
        skeleton = skeletonize(object_mask)

        # Get all skeleton coordinates
        skeleton_coords = np.argwhere(skeleton)

        if len(skeleton_coords) < 2:
            ar_object_lengths_km.append(0)
            continue

        # Compute max geodesic distance between all pairs of skeleton points
        max_dist = 0
        for i in range(len(skeleton_coords)):
            r1, c1 = skeleton_coords[i]
            lat1, lon1 = lat_grid[r1, c1], lon_grid[r1, c1]

            for j in range(i + 1, len(skeleton_coords)):
                r2, c2 = skeleton_coords[j]
                lat2, lon2 = lat_grid[r2, c2], lon_grid[r2, c2]

                dist = geodesic((lat1, lon1), (lat2, lon2)).kilometers
                if dist > max_dist:
                    max_dist = dist

        ar_object_lengths_km.append(max_dist)

    return ar_object_lengths_km

ar_object_lengths_km=calculate_ar_lengths(labelled_mask,num_objects,lats,lons,size_filter)





# --------------------------------------------------------------------
 # ---------------------- Calculate Surface Areas ---------------------
 # --------------------------------------------------------------------
# Surface area is in square kilometres.
object_area_list = []

# Compute grid cell area using latitude bounds (assuming uniform grid spacing)
R = 6371  # Earth radius in km
lat_res = np.abs(lats[1, 0] - lats[0, 0])  # Latitude resolution
lon_res = np.abs(lons[0, 1] - lons[0, 0])  # Longitude resolution

# Approximate grid cell area using spherical Earth model
grid_areas = (R**2) * np.radians(lat_res) * np.radians(lon_res) * np.cos(np.radians(lats))

object_area_list = []

for obj_label in range(1, num_objects + 1):  # Labels start at 1
    area = ndimage.sum(grid_areas[0:221,:], labels=labelled_mask, index=obj_label)
    object_area_list.append(area)

# --------------------------------------------------------------------
# -------------------------- Calculate Widths ------------------------
# --------------------------------------------------------------------
# The Width of an Object is calculated as its surface area divided
# by its length.
object_width_list = np.array(object_area_list) / np.array(ar_object_lengths_km)
for j in range(num_objects):
    if(object_width_list[j]==np.inf):
        object_width_list[j]=0
object_width_list = object_width_list.tolist()


testivt = ivt_today_field.copy().astype(float)  # Ensure it's float type to support NaNs
testivt[object_mask_new == False] = np.nan    


#------------------------------------------------------------------
#-----------------Length-width-----------------------------------
#---------------------------------------------------------------

# for j in range(num_objects):
#     if(object_width_list[j]<ar_object_lengths_km[j]):
#         continue
#     else:
#         object_width_list[j],ar_object_lengths_km[j]=ar_object_lengths_km[j],object_width_list[j]


# --------------------------------------------------------------------
# -------------------------- AR Criteria -----------------------------
# --------------------------------------------------------------------
# -------------------- Criterion 1: Length Check ---------------------
print('Length Criterion')
# Filter Objects based on axis length.
min_length = 1000
filter_list_1 = []
for j in range(num_objects):
    if ((j+1) not in size_filter):
        #& ((j+1) not in landfall_filter)):
        length_check = (ar_object_lengths_km[j] > min_length)
        if length_check == False:
            filter_list_1.append(j+1)



# --------------------------------------------------------------------

 # ------------------ Criterion 2: Narrowness Check -------------------
# Filter Objects based on length/width ratio.
print('Narrowness Criterion')
length_width_list = np.array(ar_object_lengths_km)/np.array(object_width_list)
length_width_list = length_width_list.tolist()

filter_list_2=[]
for j in range(num_objects):
    if(length_width_list[j]<2):
        filter_list_2.append(j+1)
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# ------------ Create List of Object Mean IVT Magnitude --------------
mean_ivt_magnitude_list = []

mean_ivt_magnitude_list = ndimage.mean(ivt_today_field, 
                       labelled_mask, 
                       list(range(1, num_objects+1)))



# --------------------------------------------------------------------
# ------------ Create List of Object Mean IVT Direction --------------

mean_northward_ivt = ndimage.mean(northward_ivt, 
                                  labelled_mask, 
                                  list(range(1, num_objects+1)))
mean_eastward_ivt = ndimage.mean(eastward_ivt, 
                                 labelled_mask, 
                                 list(range(1, num_objects+1)))
mean_ivt_direction_list = ((np.arctan2(mean_eastward_ivt, mean_northward_ivt) 
            * 180 / np.pi) ) % 360




#----------------------Meridonial IVT check-------------------------#
#-------------------------------------------------------------------#

print('Meridional IVT Criterion')
filter_list_3 = []
mean_poleward_ivt=[]
temp=[]
ids=[]
for j in range(num_objects):
    if (((j+1) not in size_filter) 
        #& ((j+1) not in landfall_filter) 
        & ((j+1) not in filter_list_1)):
        ids.append(j+1)
        mean_poleward_ivt=(ndimage.mean(northward_ivt, 
                                       labelled_mask, 
                                        j+1))
        temp.append(mean_poleward_ivt)
        mean_meridional_ivt_check = mean_poleward_ivt> 50
        if mean_meridional_ivt_check == False:
            filter_list_3.append(j+1)
            
               

#----------------------------------------------------------------------------#
#---------------------------------AR Snapshot--------------------------------#
#-----------------------------------------------------------------------------#

import os
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
import numpy as np
from netCDF4 import Dataset, num2date

print('Saving Data')

TIVT = np.zeros(num_objects)
Count = 0


def is_object_in_bay_but_not_arabian(j):
    """Returns True if the object is only in Bay of Bengal (lat < 0, lon > 80) and not extending to Arabian Sea (lon < 70)"""
    # Core point
    xp, yp = max_ivt_coords_list[j]
    lat_core = lats[xp, yp]
    lon_core = lons[xp, yp]
    
    # Check if core point is in Bay of Bengal
    if lat_core < 0 or lon_core > 80:
        # Check if object spans to Arabian Sea
        obj_mask = (labelled_mask == (j + 1))
        obj_lons = lons[obj_mask]
        obj_lats = lats[obj_mask]
        if np.any(obj_lons < 70) and np.all(obj_lats > -10):
            return False  # It touches Arabian Sea
        else:
            return True   # Only in Bay of Bengal
    return False  # Not in Bay of Bengal

def plot_ar_snapshot(j):
    global Count
    Count += 1

    # Estimate AR center coordinates (used in title or centering)
    lat1 = max_lat - (res * np.abs(max_ivt_coords_list[j][0]))
    lon1 = min_lon + (res * np.abs(max_ivt_coords_list[j][1]))
    AR_coords = (lat1, lon1)

    # Setup figure
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    fig.suptitle(date_text, fontsize=16, fontweight='bold')
    step = 5
    u_plot = u_today[::step, ::step]
    v_plot = v_today[::step, ::step]
    Lon_plot = lons[::step, ::step]
    Lat_plot = lats[::step, ::step]

    ax.quiver(Lon_plot, Lat_plot, u_plot, v_plot,
              scale=700, width=0.002, headlength=5, headwidth=4,
              transform=ccrs.PlateCarree(), zorder=20)

    # # Add coastlines
    ax.coastlines(resolution='50m', color='black', linewidth=1.5)

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = True
    gl.xlabel_style = {'size': 12, 'rotation': 0}
    gl.ylabel_style = {'size': 12}
    gl.xformatter = ccrs.cartopy.mpl.gridliner.LONGITUDE_FORMATTER
    gl.yformatter = ccrs.cartopy.mpl.gridliner.LATITUDE_FORMATTER

    # Axis labels
    ax.set_xlabel('Longitude', fontsize=14, labelpad=10)
    ax.set_ylabel('Latitude', fontsize=14, labelpad=10)

    # Dim all IVT values except the current object
    object_mask = (labelled_mask == (j + 1))
    ivt_isolated = np.zeros_like(ivt_today_field)
    ivt_isolated[object_mask] = ivt_today_field[object_mask]
    ivt_object = ivt_today_field[object_mask]
    # Calculate integer levels covering the range of IVT values
    ivt_min = int(np.floor(np.min(ivt_object)))
    ivt_max = int(np.ceil(np.max(ivt_object)))
    levels = np.arange(ivt_min, ivt_max + 1)  # Integer steps from min to max
    
    ivt_contour = ax.contourf(lons[0:221, :], lats[0:221, :], ivt_isolated,
                              levels=levels,  # Use integer levels
                              cmap=matplotlib.cm.get_cmap('Blues'),
                              zorder=0, extend='max')

    cbar = plt.colorbar(ivt_contour, ax=ax, orientation='vertical', shrink=0.7, pad=0.03)
    cbar.set_label('IVT (kg/m/s)', rotation=270, labelpad=15, fontsize=13, fontweight='bold')
    
    # Set integer ticks on colorbar
    cbar.locator = matplotlib.ticker.MaxNLocator(integer=True)  # Force integer ticks
    cbar.update_ticks()
    # Plot object boundary (green)
    ax.contour(lons[0:221, :], lats[0:221, :], object_mask,
               levels=[0.5], colors='green',
               linewidths=1.5, zorder=25,
               transform=ccrs.PlateCarree())
    axis_mask = (axis_cube == (j + 1))
    ax.contour(lons[0:221, :], lats[0:221, :], axis_mask,
               levels=[0.5], colors='yellow',
               linewidths=1.2, zorder=26,
               transform=ccrs.PlateCarree())

    maximum_point = max_ivt_coords_list[j]
    from geopy.distance import geodesic

    # Core point (indices)
    xp, yp = maximum_point
    lat_core = lats[xp, yp]
    lon_core = lons[xp, yp]
    core_point = (lat_core, lon_core)
    
    # Extract land points where land_mask == 1
    land_y, land_x = np.where(land_mask == 1)
    land_lats = lats[land_y, land_x]
    land_lons = lons[land_y, land_x]
        
    # Compute distance to all land points
    min_distance = float('inf')
    #all_distances=[]
    for lat, lon in zip(land_lats, land_lons):
        dist = geodesic(core_point, (lat, lon)).kilometers
        #all_distances.append(dist)
        if dist < min_distance:
            min_distance = dist
    
    distance_from_land_km = round(min_distance)


    # Choose two distinct points from the axis to define direction
    if len(axis_coords_list[j]) >= 2:
        axis_line_p1 = axis_coords_list[j][0]
        axis_line_p2 = axis_coords_list[j][1]
    else:
        return  # Skip if not enough points to define a line

    x1, y1 = axis_line_p1[0], axis_line_p1[1]
    x2, y2 = axis_line_p2[0], axis_line_p2[1]
    xp, yp = maximum_point[0], maximum_point[1]

    # Compute slope of original axis line
    dx = x2 - x1
    dy = y2 - y1

    # Get bounding box of AR object from its mask
    current_mask = (labelled_mask == (j + 1))
    rows, cols = np.where(current_mask)

    if len(rows) == 0:
        return  # No object pixels found

    i_min, i_max = rows.min(), rows.max()
    j_min, j_max = cols.min(), cols.max()

    # Determine perpendicular line endpoints, constrained to AR bounding box
    if dx == 0:
        # Axis is vertical -> Perpendicular is horizontal
        x_perp = np.array([xp - 1, xp + 1])
        y_perp = np.array([yp, yp])
        x_perp = np.clip(x_perp, i_min, i_max)
    elif dy == 0:
        # Axis is horizontal -> Perpendicular is vertical
        x_perp = np.array([xp, xp])
        y_perp = np.array([yp - 1, yp + 1])
        y_perp = np.clip(y_perp, j_min, j_max)
    else:
        m = dy / dx
        m_perp = -1 / m

        # Define x range for perpendicular line within bounding box
        x_span = np.linspace(i_min, i_max, 100)
        y_span = m_perp * (x_span - xp) + yp

        # Clip y values to AR bounding box
        inside_mask = (y_span >= j_min) & (y_span <= j_max)
        x_perp = x_span[inside_mask]
        y_perp = y_span[inside_mask]

    # Convert from index to lat/lon
    def idx_to_latlon(i, j):
        lat = max_lat - (res * i)
        lon = min_lon + (res * j)
        return lat, lon

    lat_perp = []
    lon_perp = []
    for xi, yi in zip(x_perp, y_perp):
        lat, lon = idx_to_latlon(xi, yi)
        lat_perp.append(lat)
        lon_perp.append(lon)

    # Step 1: Round to nearest grid indices (and ensure they're valid)
    x_perp_idx = np.clip(np.round(x_perp).astype(int), 0, ivt_today_field.shape[0] - 1)
    y_perp_idx = np.clip(np.round(y_perp).astype(int), 0, ivt_today_field.shape[1] - 1)

    # Step 2: Extract IVT values along the perpendicular
    ivt_values = ivt_today_field[x_perp_idx, y_perp_idx]

    # Step 3: Compute sum (or mean if preferred)
    ivt_sum_perpendicular = np.sum(ivt_values)  # total IVT along the perpendicular
    TIVT[j] = ivt_sum_perpendicular * 1000

    # Metadata for text box
    length = round(ar_object_lengths_km[j])
    width = round(object_width_list[j])
    mean_ivt_magnitude = round(mean_ivt_magnitude_list[j])
    mean_ivt_direction = round(mean_ivt_direction_list[j])
    landfall_ivt_magnitude = str(round(landfall_ivt_magnitudes[j]))
    landfall_ivt_direction = str(round(landfall_ivt_directions[j]))
    tivt_req = round(TIVT[j] * width)
    
    core_value=ivt_isolated[xp,yp]
    
    if lon_core > 72:
        textstr = (
            f"Date: {date_text}\n"
            f"Length: {length} km, Width: {width} km\n"
            f"Mean IVT Magnitude: {mean_ivt_magnitude} kg/m/s, "
            f"Mean IVT Direction: {mean_ivt_direction}°\n"
            f"Core value of the AR:{round(core_value)} kg/m/s \n"
            f"Object Boundary: Green, Axis: Yellow \n"
            f"TIVT = {tivt_req:.2e} kg/m/s, \n"
        )
        
    else:
        textstr = (
            f"Date: {date_text}\n"
            f"Length: {length} km, Width: {width} km\n"
            f"Mean IVT Magnitude: {mean_ivt_magnitude} kg/m/s, "
            f"Mean IVT Direction: {mean_ivt_direction}°\n"
            f"Core value of the AR:{round(core_value)} kg/m/s \n"
            f"Object Boundary: Green, Axis: Yellow \n"
            f"TIVT = {tivt_req:.2e} kg/m/s, \n"
            f"Distance of the core from the land = {distance_from_land_km} km "
        )
        

    # Adjust layout to create more space at the bottom
    plt.subplots_adjust(left=0.12, right=0.9, bottom=0.28, top=0.93)

    # Y-axis label
    fig.text(0.2, 0.6, 'Latitude(N)', va='center', rotation='vertical', fontsize=14)

    # X-axis label
    fig.text(0.5, 0.22, 'Longitude(E)', ha='center', fontsize=14)

    # Text box below everything
    fig.text(0.5, 0.03, textstr, ha='center', va='bottom',
             fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
 
    
    output_dir = os.path.join(datadir2, "ar_snapshots")
    os.makedirs(output_dir, exist_ok=True)
    from osgeo import gdal, osr

    filename = f"ar_{date_text}-snapshot_({Count}).tif"
    output_path = os.path.join(output_dir, filename)
    
    # Step 1: Define the lat/lon extent of the ivt_isolated grid
    min_lat1 = np.min(lats)
    max_lat1 = np.max(lats)
    min_lon1 = np.min(lons)
    max_lon1 = np.max(lons)
    
    # Step 2: Define pixel size
    nrows, ncols = ivt_isolated.shape
    pixel_width = (max_lon1 - min_lon1) / ncols
    pixel_height = (max_lat1 - min_lat1) / nrows
    
    # Step 3: Set geotransform
    # Format: (top-left lon, pixel width, 0, top-left lat, 0, -pixel height)
    geotransform = (min_lon1, pixel_width, 0, max_lat1, 0, -pixel_height)
    
   # Step 4: Set spatial reference (WGS84)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    
    
    
    # Step 5: Save to GeoTIFF
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(output_path, ncols, nrows, 1, gdal.GDT_Float32)
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(np.flipud(ivt_isolated))  # Flip array vertically
    ds.GetRasterBand(1).SetNoDataValue(0)
    ds.FlushCache()
    ds = None



if current_month in [10, 11, 12, 1]:
    if num_objects > 0:
        for j in range(num_objects):
            if ((j + 1) not in size_filter) and not is_object_in_bay_but_not_arabian(j):
                plot_ar_snapshot(j)

else:
    if num_objects > 0:
        for j in range(num_objects):
            obj_id = j + 1
            if (obj_id not in size_filter and 
                obj_id not in filter_list_1 and 
                obj_id not in filter_list_2 and 
                obj_id not in filter_list_3 and 
                not is_object_in_bay_but_not_arabian(j)):
                plot_ar_snapshot(j)



