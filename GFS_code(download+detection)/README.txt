Instructions for running AR Detection Algorithm for GFS data

📁 Files Included:
Python script for AR detection
Land mask file (.mat format)

------------------------------------------------------------------------------------------------------
⚙️ Required Setup:
You will need to modify three directory paths in the script before execution:

datadir – This is your input directory, where the GFS data files will go after getting downloaded.

output_dir- This is the directory where the GFS IVT Plots will get saved

datadir2 – This is the output directory, where the script will save the generated GeoTIFF images of the detected Atmospheric Rivers.

land_mask_dir – Path to the land mask .mat file (included in the ZIP). Ensure that this file is correctly referenced.

---------------------------------------------------------------------------------------------------------

Once the paths are updated, the script is ready to run. It will output:

Detected AR object images (GeoTIFF format)
AR characteristics

--------------------------------------------------------------------------------------------------------

Note: Make sure you have all the python libraries required in the code installed before running the python script.
