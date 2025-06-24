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


# --------------------------------------------------------------------
# ------------------------ Define Parameters -------------------------
# --------------------------------------------------------------------

datadir = r'C:\Users\patti\OneDrive\Desktop\ISRO\IVT_calculation\All_datasets(ERA5)\Testdata13'
os.chdir(datadir)
datadir2=r'C:\Users\patti\OneDrive\Desktop\ISRO\ARs'


selected_files = [f for f in os.listdir(datadir) if f.endswith('.nc')]


def extract_date(filename):
    # Adjust this depending on your exact filename pattern
    base = os.path.splitext(filename)[0]  # removes .nc
    date_part = base.split('_')[-1]       # e.g., '2021-11-01'
    return datetime.strptime(date_part, '%d-%m-%Y')

selected_files_sorted = sorted(selected_files, key=extract_date)


min_ivt = 150  # IVT threshold fixed lower limit (kg m*-1 s*-1)
threshold_percentage = 85  # Defines the IVT percentile threshold
min_length = 2000  # Minimum length of ARs in km
min_aspect = 2  # Minimum length/width ratio
min_span = 1000 # Minimum required distance between AR axis start/end points

restrict_landfalling = 0
min_size = 60 # Minimum Size - Number of Grid Cells
# 

# Load Landfall Domain
land_mask_dir = r'C:\Users\patti\OneDrive\Desktop\gujarat_to_kerala_coast_mask.mat' # Names land mask file
land_mask = loadmat(land_mask_dir)
wc=list(land_mask.items())[3]
land_mask=wc[1]
land_mask=np.flipud(land_mask)


# Enter the grid resolution of the data in degrees
res = 0.25
# Ener the input data lat/lon extent
max_lat = 40
min_lat = -15
max_lon = 100
min_lon = 50
# Enter the number of latitude/longitude grid cells of the input .nc files
lon_grid_number = 201
lat_grid_number = 221

pre=[500,550,600,650,700,750,775,800,825,850,875,900,925,950,975,1000]

last_10_days_ivtx = np.full((1, 221, 201), np.nan)  # Initialize with NaNs

last_10_days_ivty = np.full((1, 221, 201), np.nan)
# Run your loop over selected files
for fname in selected_files_sorted:
    print(fname)
# for i in range(158,168):#len(files)):
#     fname=files[i]
#     print(fname)
    nc=Dataset(fname,'r')
    lon = nc.variables['longitude'][:].data
    lat = nc.variables['latitude'][:].data
    
    # Extract data for q, u, v (dim: time, level, lat, lon)
    q = nc.variables['q'][:].data
    u = nc.variables['u'][:].data
    v = nc.variables['v'][:].data
    
    time_var=nc.variables['valid_time']
    time= num2date(time_var[:], units=time_var.units).data
    lis=[]
    
    
    for idx, cur_time in enumerate(time):
         cur_time_str = cur_time.strftime("%Y-%m-%d %H:%M UTC")
         cur_date_str = cur_time.strftime("%Y-%m-%d")
         lis.append(cur_time_str)
    
    lat0=lat.astype(np.float64)
    lon0=lon.astype(np.float64)
    lonn,latt=np.meshgrid(lon0,lat0)
    del lon0, lat0
    
    # Pre-calculate pressure level differences
    dp = np.diff(pre)  # Assuming pre is the array of pressure levels
    
   
    # Array to store the last 5 days (rolling window)
    
    # Now extract u and v at 850 hPa for each hour
    dailyu850 = []
    dailyv850 = []
    # Loop over days (24-hour intervals)
    
        # Extract data for the 24 hours
        # print(tt)
    reqQ = q[:, :, 0:221, 0:201]  
    reqU = u[:, :, 0:221, 0:201]
    reqV = v[:, :, 0:221, 0:201]
        
        # Calculate IVT for each hour
    hourly_ivt_sumx = []
    hourly_ivt_sumy = []
    hourly_ivt_sum = []
    hourly_sumu = []
    hourly_sumv = []
        
    for hr in range(0,24):  # Loop over hours
        # print(hr)
            # Compute averages for each pressure layer
        qavg = (reqQ[hr, :-1, :, :] + reqQ[hr, 1:, :, :]) / 2  
        uavg = (reqU[hr, :-1, :, :] + reqU[hr, 1:, :, :]) / 2
        vavg = (reqV[hr, :-1, :, :] + reqV[hr, 1:, :, :]) / 2
        u850 = reqU[hr, 6, :, :]  # shape (24, 201, 201)
        v850 = reqV[hr, 6, :, :]  # shape (24, 201, 201)
        hourly_sumu.append(u850)
        hourly_sumv.append(v850)
        
            # Calculate IVT for all pressure levels
        ivt = (1 / 9.8) * np.sqrt((qavg * uavg * dp[:, None, None] * 100)**2 + 
                                       (qavg * vavg * dp[:, None, None] * 100)**2)
        ivtx = (1 / 9.8) * (qavg * uavg * dp[:, None, None] * 100)
                                       
        ivty = (1 / 9.8) * (qavg * vavg * dp[:, None, None] * 100)
            
            # Sum along the 27th dimension (pressure levels)
        ivt_sum = np.sum(ivt, axis=0)  
        ivt_sumx = np.sum(ivtx, axis=0)
        ivt_sumy = np.sum(ivty, axis=0)
        hourly_ivt_sum.append(ivt_sum)
        hourly_ivt_sumx.append(ivt_sumx)
        hourly_ivt_sumy.append(ivt_sumy)
        
        # Mean along the 24 hours (time steps) for the current day
    daily_mean = np.mean(hourly_ivt_sum, axis=0) 
    daily_meanx = np.mean(hourly_ivt_sumx, axis=0)
    daily_meany = np.mean(hourly_ivt_sumy, axis=0)
    dailyu850 = np.mean(np.array(hourly_sumu), axis=0) 
    dailyv850 = np.mean(np.array(hourly_sumv), axis=0)
        
        # Store last 5 days of IVTx and IVTy
    last_10_days_ivtx[:-1] = last_10_days_ivtx[1:]  # Shift array up
    last_10_days_ivtx[-1] = daily_meanx  # Store current day's IVTx at the last index
    last_10_days_ivty[:-1] = last_10_days_ivty[1:]  # Shift array up
    last_10_days_ivty[-1] = daily_meany
       
       
northward_ivt = daily_meany.copy()
eastward_ivt = daily_meanx.copy()
ivt = daily_mean.copy()
#zero = 0 * ivt # Create zero cube with same shape as IVT data
    # Compute ivt direction within each cell
ivt_direction = ((np.arctan2(eastward_ivt.data, northward_ivt.data) 
                           * 180 / np.pi) + 180) % 360
northward_ivt_0 = last_10_days_ivty.copy()
eastward_ivt_0 = last_10_days_ivtx.copy()
    
ivt_0 = np.power(np.power(northward_ivt_0, 2)
                   + np.power(eastward_ivt_0, 2), 1/2)
ivt_0 = np.nanmean(ivt_0, axis=0)
#zero_0 = 0 * ivt_0
    # Compute IVT percentile threshold
ivt_0_masked = ivt_0.copy()

current_month = cur_time.month  # Get the month from the last timestamp
current_month = cur_time.month  # Get the month from the last timestamp
if current_month in [10, 11, 12, 1]:  # October to January
    ivt[ivt > 600] = np.nan
    ivt_0[ivt_0 > 600] = np.nan
    
    
    ivt_0[129:,:]=np.nan

    ivt_percentile_threshold = np.nanpercentile(ivt_0, threshold_percentage)
    ivt_lower_limit = min_ivt
    ivt_threshold = np.maximum(ivt_percentile_threshold, ivt_lower_limit)
    object_mask = (ivt > ivt_threshold)

    # mask_data = np.load('C:\\Users\patti\OneDrive\Desktop\ISRO\IVT_calculation\Own codes\mask_grid_with_coords.npz')
    # mask_grid = mask_data['mask_grid']     # Shape: (221, 201)
    # mask_lats = mask_data['lats']
    # mask_lons = mask_data['lons']

    # # Safety check
    # assert object_mask.shape == mask_grid.shape, "Shape mismatch between IVT field and mask grid!"

    # ivt_lats = mask_lats  
    # mask_grid = np.flipud(mask_grid)

    # labelled_mask, num_objects = ndimage.label(object_mask)

    # valid_object_labels = []
    # for obj_id in range(1, num_objects + 1):
    #     obj_mask = (labelled_mask == obj_id)
    #     if np.any(mask_grid[obj_mask] == 1):  # Object intersects shapefile region
    #         valid_object_labels.append(obj_id)

    # # Create filtered mask keeping only intersecting objects
    # filtered_object_mask = np.isin(labelled_mask, valid_object_labels)

    # # Update object mask and re-label
    # labelled_mask, num_objects = ndimage.label(filtered_object_mask)
    # object_mask = filtered_object_mask.copy()

    # Mask where the object exists
    # object_mask = (labelled_mask == (j + 1))
    
    # Create mask where both eastward and northward IVT are positive within the object
    eivt_positive = eastward_ivt > 0
    nivt_positive = northward_ivt > 0
    
    # Combine conditions: within object AND both components positive
    combined_mask = object_mask & (eivt_positive | nivt_positive)
    combined_mask[129:,:]=False
    object_mask=combined_mask
    
    
else:
    ivt_percentile_threshold = np.nanpercentile(ivt_0, threshold_percentage)
    ivt_lower_limit = min_ivt
    ivt_threshold = np.maximum(ivt_percentile_threshold, ivt_lower_limit)
    object_mask=ivt>ivt_threshold
    



# --------------------------------------------------------------------
# ------------------------ Identify Objects and landfall --------------------------

landfall_filter = []
landfall_filter = land_mask[0:221,:]*object_mask

# --------------------------------------------------------------------
# --------------------------- Size Filter ----------------------------
# Filter small Objects to speed up code run time.
from scipy.ndimage import label, sum as nd_sum
import numpy as np

size_filter = []

# Label the connected AR components
labelled_mask, num_objects = label(object_mask)

# Compute the size (number of grid cells) of each object
object_sizes = nd_sum(object_mask, labelled_mask, index=range(1, num_objects + 1))

# --- Correct landfall filter creation ---
# Get object IDs that intersect land
landfall_ids = np.unique(labelled_mask[land_mask == 1])
landfall_ids = landfall_ids[landfall_ids > 0]  # Remove background (0)
landfall_filter = landfall_ids.tolist()

# --- Apply filtering ---
size_filter = []

for j in range(num_objects):  # j from 0 to num_objects-1, corresponds to object ID j+1
    obj_id = j + 1
    if obj_id not in landfall_filter:
        if object_sizes[j] <= min_size:
            size_filter.append(obj_id)


"""
unique_values = np.unique(landfall_filter)
print(unique_values)
"""

# --------------------------------------------------------------------
 # -------------------------- Compute Axis ----------------------------
 # -------------------------------------------------------------------
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
    obj_id = j+1

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
    object_ivt = object_mask_new * ivt

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
        landfall_ivt_directions.append(ivt_direction[row, col])
    else:
        landfall_locations.append((0, 0))
        landfall_ivt_magnitudes.append(0)
        landfall_ivt_directions.append(0)

    if (row, col) != (0, 0):
        ivt_copy_forward = object_ivt.copy()
        ivt_copy_backward = object_ivt.copy()

        forward_axis, forward_coords = trace_axis(ivt_copy_forward, ivt_direction, row, col, reverse=False)
        backward_axis, backward_coords = trace_axis(ivt_copy_backward, ivt_direction, row, col, reverse=True)

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
        step_landfall_ivt_directions.append(ivt_direction[axis_coords[-1][0], axis_coords[-1][1]])
    else:
        step_landfall_ivt_directions.append(0)




#---------------------------AR_Object Length Computation--------------------------#
from skimage.morphology import skeletonize
from geopy.distance import geodesic
import numpy as np

def calculate_ar_lengths(labelled_mask, num_objects, lat, lon, size_filter):
    # Generate 2D lat-lon grid
    lon_grid, lat_grid = np.meshgrid(lon, lat)

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




ar_object_lengths_km = calculate_ar_lengths(labelled_mask, num_objects, lat, lon, size_filter)



"""
unique_vals=np.unique(axis_length_list)
print(unique_vals)

"""

 # --------------------------------------------------------------------
 # ---------------------- Calculate Surface Areas ---------------------
 # --------------------------------------------------------------------
# Surface area is in square kilometres.
object_area_list = []

# Compute grid cell area using latitude bounds (assuming uniform grid spacing)
R = 6371  # Earth radius in km
lat_res = np.abs(latt[1, 0] - latt[0, 0])  # Latitude resolution
lon_res = np.abs(lonn[0, 1] - lonn[0, 0])  # Longitude resolution

# Approximate grid cell area using spherical Earth model
grid_areas = (R**2) * np.radians(lat_res) * np.radians(lon_res) * np.cos(np.radians(latt))

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
object_width_list = object_width_list.tolist()



# testivt = ivt.copy().astype(float)  # Ensure it's float type to support NaNs
# testivt[object_mask_new == False] = np.nan    
    



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
length_width_list = np.array(ar_object_lengths_km) / np.array(object_width_list)
length_width_list = length_width_list.tolist()

filter_list_2=[]
for j in range(num_objects):
    if(length_width_list[j]<2):
        filter_list_2.append(j+1)
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# ------------ Create List of Object Mean IVT Magnitude --------------
mean_ivt_magnitude_list = []

mean_ivt_magnitude_list = ndimage.mean(ivt, 
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






# --------------------------------------------------------------------

# --------------------------------------------------------------------
#-------------- Criterion 3: Mean Meridional IVT Check ---------------
# An object is discarded if the mean IVT does not have a poleward 
# component > 50 kg m*-1 s*-1.
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
            
       
        
       
#----------------------------------------------------------------------#
#-------------------------------Create AR Snapshot---------------------#
#----------------------------------------------------------------------#
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
    lat_core = latt[xp, yp]
    lon_core = lonn[xp, yp]
    
    # Check if core point is in Bay of Bengal
    if lat_core < 0 or lon_core > 80:
        # Check if object spans to Arabian Sea
        obj_mask = (labelled_mask == (j + 1))
        obj_lons = lonn[obj_mask]
        obj_lats = latt[obj_mask]
        if np.any(obj_lons < 70) and np.all(obj_lats > 0):
            return False  # It touches Arabian Sea
        else:
            return True   # Only in Bay of Bengal
    return False  # Not in Bay of Bengal


def plot_ar_snapshot(j):
    global Count
    Count += 1

    # Extract date and time from file
    nc = Dataset(fname, 'r')
    date_time_var = nc.variables['valid_time']
    valid_time = num2date(date_time_var[0], units=date_time_var.units,
                          calendar=getattr(date_time_var, 'calendar', 'standard'))
    date_text = valid_time.strftime('%Y-%m-%d')
    time_text = valid_time.strftime('%H')

    # Estimate AR center coordinates (used in title or centering)
    lat1 = max_lat - (res * np.abs(max_ivt_coords_list[j][0]))
    lon1 = min_lon + (res * np.abs(max_ivt_coords_list[j][1]))
    AR_coords = (lat1, lon1)

    # Setup figure
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    fig.suptitle(date_text, fontsize=16, fontweight='bold')
    step = 5
    u_plot = dailyu850[::step, ::step]
    v_plot = dailyv850[::step, ::step]
    Lon_plot = lonn[::step, ::step]
    Lat_plot = latt[::step, ::step]

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
    ivt_isolated = np.zeros_like(ivt)
    ivt_isolated[object_mask] = ivt[object_mask]
    ivt_object = ivt[object_mask]

    # Plot dimmed IVT background, with the current object highlighted
    ivt_min = int(np.floor(np.min(ivt_object)))
    ivt_max = int(np.ceil(np.max(ivt_object)))
    levels = np.arange(ivt_min, ivt_max + 1)  # Integer steps from min to max   
    ivt_contour = ax.contourf(lonn[0:221, :], latt[0:221, :], ivt_isolated,
                              levels=levels,  # Use integer levels
                              cmap=matplotlib.cm.get_cmap('Blues'),
                              zorder=0, extend='max')

    cbar = plt.colorbar(ivt_contour, ax=ax, orientation='vertical', shrink=0.7, pad=0.03)
    cbar.set_label('IVT (kg/m/s)', rotation=270, labelpad=15, fontsize=13, fontweight='bold')
    
    # Set integer ticks on colorbar
    cbar.locator = matplotlib.ticker.MaxNLocator(integer=True)  # Force integer ticks
    cbar.update_ticks()

    # Plot object boundary (green)
    ax.contour(lonn[0:221, :], latt[0:221, :], object_mask,
               levels=[0.5], colors='green',
               linewidths=1.5, zorder=25,
               transform=ccrs.PlateCarree())
    axis_mask = (axis_cube == (j + 1))
    ax.contour(lonn[0:221, :], latt[0:221, :], axis_mask,
               levels=[0.5], colors='yellow',
               linewidths=1.2, zorder=26,
               transform=ccrs.PlateCarree())

    maximum_point = max_ivt_coords_list[j]
    from geopy.distance import geodesic

    # Core point (indices)
    xp, yp = maximum_point
    lat_core = latt[xp, yp]
    lon_core = lonn[xp, yp]
    core_point = (lat_core, lon_core)
    
    # Extract land points where land_mask == 1
    land_y, land_x = np.where(land_mask == 1)
    land_lats = latt[land_y, land_x]
    land_lons = lonn[land_y, land_x]
        
    # Compute distance to all land points
    min_distance = float('inf')
    for lat, lon in zip(land_lats, land_lons):
        dist = geodesic(core_point, (lat, lon)).kilometers
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
    x_perp_idx = np.clip(np.round(x_perp).astype(int), 0, ivt.shape[0] - 1)
    y_perp_idx = np.clip(np.round(y_perp).astype(int), 0, ivt.shape[1] - 1)

    # Step 2: Extract IVT values along the perpendicular
    ivt_values = ivt[x_perp_idx, y_perp_idx]

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

    # Save output
    output_dir = os.path.join(datadir2, "ar_snapshots")
    os.makedirs(output_dir, exist_ok=True)
    from osgeo import gdal, osr

    filename = f"ar_{date_text}-{time_text}-snapshot_({Count}).tif"
    output_path = os.path.join(output_dir, filename)
    
    # Step 1: Define the lat/lon extent of the ivt_isolated grid
    min_lat1 = np.min(latt)
    max_lat1 = np.max(latt)
    min_lon1 = np.min(lonn)
    max_lon1 = np.max(lonn)
    
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
    ds.GetRasterBand(1).WriteArray(ivt_isolated)
    ds.GetRasterBand(1).SetNoDataValue(0)
    ds.FlushCache()
    ds = None  # Close file



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

