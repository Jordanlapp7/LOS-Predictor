from google.cloud import bigquery
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

os.environ["GCLOUD_PROJECT"] = os.getenv("GCLOUD_PROJECT")

# Load BigQuery Client
client = bigquery.Client()

# SQL Query
query = """
SELECT 
    pat.subject_id,
    adm.hadm_id,
    pat.gender,
    pat.anchor_age AS age,
    adm.race,
    adm.admission_type,
    adm.insurance,
    adm.admittime,
    adm.dischtime,
    EXTRACT(DAY FROM adm.dischtime - adm.admittime) AS length_of_stay,
    diag.icd_code AS primary_diagnosis
FROM `physionet-data.mimiciv_3_1_hosp.patients` AS pat
JOIN `physionet-data.mimiciv_3_1_hosp.admissions` AS adm 
    ON pat.subject_id = adm.subject_id
JOIN `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` AS diag
    ON adm.hadm_id = diag.hadm_id
WHERE adm.dischtime IS NOT NULL
"""

def load_data():
    """Fetch data from BigQuery and return as DataFrame"""
    df = client.query(query).to_dataframe()
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())