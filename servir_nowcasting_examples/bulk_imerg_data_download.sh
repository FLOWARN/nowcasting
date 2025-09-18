#!/bin/bash
#SBATCH --job-name=bulk_imerg_download
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=gpu2
#SBATCH --output=/home1/ppatel2025/ppworktp/final/nowcasting/servir_nowcasting_examples/logs/bulk_imerg_%j.out
#SBATCH --error=/home1/ppatel2025/ppworktp/final/nowcasting/servir_nowcasting_examples/logs/bulk_imerg_%j.err

# Usage: sbatch bulk_imerg_data_downloader.sh <xmin> <ymin> <xmax> <ymax> <start_datetime> <end_datetime>
# Example: sbatch bulk_imerg_data_downloader.sh -3.5 4.7 1.7 11.9 "2022-09-19 00:00:00" "2022-09-21 00:00:00"
# Data will be saved to: ./data/bulk_events/

# Load conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

conda activate tito_env

# Use the directory where the job was submitted from
SCRIPT_DIR="$SLURM_SUBMIT_DIR"
cd "$SCRIPT_DIR"

# Create logs directory in the project directory
mkdir -p "$SCRIPT_DIR/logs"

echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $SCRIPT_DIR"
echo "Downloading IMERG data from $5 to $6 for coordinates ($1,$2) to ($3,$4)"
echo "Raw data will be saved to: $SCRIPT_DIR/data/raw/"
echo "Processed H5 file will be saved to: $SCRIPT_DIR/data/processed/imerg_data.h5"

# Create Python download script in the project directory
cat > "$SCRIPT_DIR/temp_download_${SLURM_JOB_ID}.py" << 'EOF'
import sys, os, datetime as DT
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from servir.utils.download_data import get_gpm_files, initialize_event_folders
from servir.utils.m_tif2h5py import tif2h5py

# Get parameters
xmin, ymin, xmax, ymax = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
start_time = DT.datetime.strptime(sys.argv[5], '%Y-%m-%d %H:%M:%S')
end_time = DT.datetime.strptime(sys.argv[6], '%Y-%m-%d %H:%M:%S')

# Validate coordinates
if not (-180 <= xmin <= 180) or not (-180 <= xmax <= 180):
    print(f"ERROR: Longitude must be between -180 and 180. Got xmin={xmin}, xmax={xmax}")
    sys.exit(1)

if not (-90 <= ymin <= 90) or not (-90 <= ymax <= 90):
    print(f"ERROR: Latitude must be between -90 and 90. Got ymin={ymin}, ymax={ymax}")
    sys.exit(1)

if xmin >= xmax:
    print(f"ERROR: xmin ({xmin}) must be less than xmax ({xmax})")
    sys.exit(1)

if ymin >= ymax:
    print(f"ERROR: ymin ({ymin}) must be less than ymax ({ymax})")
    sys.exit(1)

if start_time >= end_time:
    print(f"ERROR: Start time must be before end time")
    sys.exit(1)

print(f"Coordinates validated: ({xmin}, {ymin}) to ({xmax}, {ymax})")
print(f"Time range validated: {start_time} to {end_time}")

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_folder = os.path.join(script_dir, 'data', 'raw') + '/'
processed_data_folder = os.path.join(script_dir, 'data', 'processed') + '/'

# Create directories
Path(raw_data_folder).mkdir(exist_ok=True, parents=True)
Path(processed_data_folder).mkdir(exist_ok=True, parents=True)

# Generate unique folder name for raw data based on timestamp
start_timestamp = sys.argv[5].replace(' ', '_').replace(':', '-')
end_timestamp = sys.argv[6].replace(' ', '_').replace(':', '-')
raw_session_folder = f"imerg_{start_timestamp}_to_{end_timestamp}/"
precipEventFolder = os.path.join(raw_data_folder, raw_session_folder)
Path(precipEventFolder).mkdir(exist_ok=True, parents=True)

print(f"Raw data folder: {precipEventFolder}")
print(f"Processed data folder: {processed_data_folder}")
# Download data
ppt_server_path = 'https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/gis/early/'
email = 'aaravamudan2014@my.fit.edu'

print("Downloading IMERG files...")
get_gpm_files(precipEventFolder, start_time, end_time, ppt_server_path, email, 360, 516)

# Convert to H5
print("Converting to H5 format...")
meta_fname = os.path.join(precipEventFolder, 'metadata.json')
h5_output = os.path.join(processed_data_folder, 'imerg_data.h5')
tif2h5py(tif_directory=precipEventFolder, h5_fname=h5_output, meta_fname=meta_fname,
         x1=xmin, y1=ymin, x2=xmax, y2=ymax)

print(f"Download complete!")
print(f"Raw TIF files: {precipEventFolder}")
print(f"Processed H5 file: {h5_output}")
EOF

# Run download
python "$SCRIPT_DIR/temp_download_${SLURM_JOB_ID}.py" "$1" "$2" "$3" "$4" "$5" "$6"

# Cleanup
rm -f "$SCRIPT_DIR/temp_download_${SLURM_JOB_ID}.py"

echo "Job completed!"