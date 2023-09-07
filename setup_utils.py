import os
from azure.storage.blob import BlobServiceClient
import pandas as pd

def fetch_data(CONNECTION_STRING):
    """fetch data from azure blob storage and save to local data folder
    """
    CONTAINER_NAME = "quarterly-results"
    BLOB_FILES = [
        "hackathon-team-simpsons-paradox/Input Data/brand_segment_mapping_hackathon.xlsx",
        "hackathon-team-simpsons-paradox/Input Data/macro_data.xlsx",
        "hackathon-team-simpsons-paradox/Input Data/maximum_discount_constraint_hackathon.xlsx",
        "hackathon-team-simpsons-paradox/Input Data/sales_data_hackathon.xlsx",
        "hackathon-team-simpsons-paradox/Input Data/volume_variation_constraint_hackathon.xlsx",
        "hackathon-team-simpsons-paradox/Submission Template/submission_template_hackathon.csv",
    ]

    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    if "data" not in [a for a in os.listdir() if os.path.isdir(a)]:
        os.mkdir("data")

    for file in BLOB_FILES:
        blob_client = container_client.get_blob_client(file)
        blob_data = blob_client.download_blob()
        with open("data/" + file.split("/")[2], "wb") as f:
            print("data/" + file.split("/")[2])
            f.write(blob_data.readall())

def load_data():
    """load data from local data folder
    """
    df_dict = {}
    for file_path in [file for file in os.listdir("data") if ".xlsx" in file]:
        for sheet_name in pd.ExcelFile("data/"+file_path).sheet_names:
            df = pd.read_excel("data/"+file_path, sheet_name=sheet_name)
            df.columns = [
                (
                    k
                    .lower()
                    .replace("_lc","")
                    .replace("pricesegment","segment")
                )
                for k in df.columns]
            dict_key = file_path.split(".")[0] + ("" if sheet_name == "Sheet1" else "_"+sheet_name.lower())
            df_dict[dict_key] = df
            

    return (
        df_dict["brand_segment_mapping_hackathon"],
        df_dict["macro_data"],
        df_dict["maximum_discount_constraint_hackathon_brand"],
        df_dict["maximum_discount_constraint_hackathon_pack"],
        df_dict["maximum_discount_constraint_hackathon_pricesegment"],
        df_dict["sales_data_hackathon"],
        df_dict["volume_variation_constraint_hackathon"]
    )
