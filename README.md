🌧️ Automated Detection of Atmospheric Rivers over India <br>
This repository contains code for detecting Atmospheric Rivers (ARs) over the Indian subcontinent using reanalysis, satellite, and forecast datasets. <br>
The project was developed as part of a research internship at NRSC, ISRO, under the mentorship of Mrs. Shivali Verma (Scientist/Engineer-SE). <br>

📌 Overview <br>
Atmospheric Rivers (ARs) are long, narrow corridors of intense moisture transport responsible for extreme precipitation and flood events. While ARs have been well studied globally, their role in India's changing climate remains underexplored.

This project presents a comprehensive AR detection pipeline customized for the Indian region. The detection framework:

Uses Integrated Vapor Transport (IVT) fields from ERA5 reanalysis and GFS forecasts

Incorporates Total Precipitable Water (TPW) from INSAT-3D/3DR and AMSR2 satellites

Identifies, filters, and characterizes AR events

Supports both retrospective and near-real-time analysis

📂 Key Features <br>
🔍 Object-based AR Detection using dynamic thresholding and IVT geometry

🗺️ AR Axis Tracing and calculation of length, width, landfall metrics

🌐 Multi-source integration: ERA5, GFS, INSAT, AMSR2

📸 Snapshot Generation with annotated AR boundaries, axis, wind vectors

🧭 Seasonal Filtering logic for Indian monsoon and winter behavior

🗃️ GeoTIFF Export for use in GIS tools (e.g., QGIS)

🛰️ Datasets Used <br>
ERA5 Reanalysis (ECMWF) – IVT, IWV

GFS Forecast Data – For forward-looking detection

INSAT-3D/3DR – High-resolution TPW over India

AMSR2 (GCOM-W1) – Oceanic TPW 

<br>

Region of interest: 50°E–100°E, -15°N–40°N, focusing on Gujarat and Kerala  


🛠️ Tools & Libraries <br>
Python (NumPy, SciPy, Matplotlib, Cartopy, NetCDF4, h5py, pygrib)

QGIS – Visualization of GeoTIFF outputs

Paramiko & FileZilla – For SFTP data access from MOSDAC (INSAT)

Anaconda + Spyder – Development environment

🧪 Methodology Highlights  

Compute IVT from wind and humidity across pressure levels

Apply percentile-based IVT threshold (e.g., 85th percentile)

Identify AR candidates using connected component labeling

Filter objects based on:

Landfall (coastal grid overlap)

Minimum size (≥60 grid cells)

Axis length (≥1000 km)

Narrowness (length/width > 2)

Poleward IVT component

Visualize snapshots with AR contours, axes, and metadata

Download & visualize TPW from INSAT & AMSR2 for comparison


🧱 Limitations  
Satellite gaps and inconsistencies (cloud cover, swath gaps)

Fixed IVT thresholds may miss events in some seasons

Data-heavy processing and dependencies on large files

🚀 Future Improvements
Integrate Machine Learning for adaptive thresholding and data fusion

Expand real-time forecasting capability with live satellite feeds

Add web-based visualization dashboard for AR tracking

📚 References
Guan & Waliser (2015, 2019) – Global AR detection methodology

NOAA, NASA Earthdata, ECMWF, IMD – Data providers

See my project report in the report section for full methodology




