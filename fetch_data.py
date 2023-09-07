import os
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from dotenv import load_dotenv
load_dotenv()

# Using Connection String
CONNECTION_STRING = os.getenv("CONNECTION_STRING")
CONTAINER_NAME = "quarterly-results"
BLOB_FILES = [
    "hackathon-team-simpsons-paradox/brand_segment_mapping_hackathon.xlsx",
    "hackathon-team-simpsons-paradox/macro_data.xlsx",
    "hackathon-team-simpsons-paradox/maximum_discount_constraint_hackathon.xlsx",
    "hackathon-team-simpsons-paradox/sales_data_hackathon_original.xlsx",
    "hackathon-team-simpsons-paradox/volume_variation_constraint_hackathon.xlsx",
]

blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

if "data" not in [a for a in os.listdir() if os.path.isdir(a)]:
    os.mkdir("data")

for file in BLOB_FILES:
    blob_client = container_client.get_blob_client(file)
    blob_data = blob_client.download_blob()
    with open("data/" + file.split("/")[1], "wb") as f:
        f.write(blob_data.readall())
