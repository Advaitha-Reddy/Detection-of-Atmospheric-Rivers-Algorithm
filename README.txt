Instructions for Running the AR Detection Script on ERA5 Data

📁 Files Included:
Python script for AR detection
Land mask file (.mat format)

------------------------------------------------------------------------------------------------------
⚙️ Required Setup:
You will need to modify three directory paths in the script before execution:

datadir – This is your input directory, where you should place the ERA5 data files (downloaded from the Copernicus Climate Data Store).

datadir2 – This is the output directory, where the script will save the generated GeoTIFF images of the detected Atmospheric Rivers.

land_mask_dir – Path to the land mask .mat file (included in the ZIP). Ensure that this file is correctly referenced.

-------------------------------------------------------------------------------------------------------
⏳ Threshold Calculation (Rolling Window):
The script uses a rolling window method to calculate the IVT threshold.

By default, it uses data from the previous 1 day.

You may customize the code to extend the rolling window—for example, use the last 10 days of IVT data to calculate a more stable threshold.

🔍 Example:
If you are detecting ARs for 12 January 2020, and using a 1-day window, download ERA5 data for 11 Jan and 12 Jan 2020.
If using a 10-day window, you’ll need data from 2 Jan to 12 Jan 2020.
---------------------------------------------------------------------------------------------------------

Once the paths are updated, the script is ready to run. It will output:

Detected AR object images (GeoTIFF format
AR characteristics

--------------------------------------------------------------------------------------------------------

Note: Make sure you have all the python libraries required in the code installed before running the python script.
