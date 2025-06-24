Instructions for Running the INSAT Data Download and TPW Plotting Script

This folder contains a Python script to download INSAT-3D data (from MOSDAC’s Standing Order folder) and plot Total Precipitable Water (TPW) over the Indian region.
-------------------------------------------------------------------------------------
✅ How It Works:
When the script runs, it prompts the user:

Enter your choice:
1 - Use most recent 3 files
2 - Select a specific date
If you enter 1:
The script assumes the current date and
Connects to the StandingOrder folder on the SFTP server.
Downloads the .h5 files.
Computes and plots the average TPW from the most recent 3 files.

If you enter 2:
The script prompts you to enter a date in DD-MM-YYYY format. It:
Finds and downloads all .h5 files for that date in StandingOrder.
Computes and plots the average TPW for the specified date.

--------------------------------------------------------------------------------
Before You Run the Script:
You must modify the following fields in the Python script:

sftp_user = 'your_username'   # Replace with your MOSDAC username
sftp_pass = 'your_password'   # Replace with your MOSDAC password
input_dir = 'DownloadedFiles' #  Folder where data will be downloaded

---------------------------------------------------------------------------------

📌 Important Note
To download data using this script:
The required INSAT-3D TPW files must first be added to your Standing Order on the MOSDAC website.

You need to:
Log in to your MOSDAC account.
Add INSAT-3D Level-2 TPW products to your Standing Order list.

Only then will the script be able to access those files for download.
---------------------------------------------------------------------------------

After modifying these things in the code, you are good to go...
