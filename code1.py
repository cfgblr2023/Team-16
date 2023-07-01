import os
import shutil
import pandas as pd

# Set the path to the CSV file
csv_file = r'C:\cfg\Issues_Collected_category and issue.csv'

# Set the source directory where the images are currently located
source_dir = r'C:\cfg\Photos_Walking Audit'

# Set the destination directory where the images will be moved to
destination_dir = r'C:\cfg\Photos_Walking'

# Load the CSV file into a pandas DataFrame
data = pd.read_csv(csv_file)

# Iterate over each row in the DataFrame
for index, row in data.iterrows():
    image_name = row['Photo']
    subclass_name = row['Category']+row['Issue type']
    image_name = image_name
    
    # Create the destination directory if it doesn't exist
    subclass_dir = os.path.join(destination_dir, subclass_name)
    os.makedirs(subclass_dir, exist_ok=True)
   
    # Set the source and destination paths
    source_path = os.path.join(source_dir, image_name)
    destination_path = os.path.join(subclass_dir, image_name)
    
    # Move the image to the destination directory
    try:
        shutil.move(source_path, destination_path)
        print(f"Moved {image_name} to {subclass_dir}")
    except FileNotFoundError:
        print(f"File {image_name} does not exist in {source_dir}")
