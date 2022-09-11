"""Requests simulation module"""

import json
import time
import random

import requests

URL = "http://127.0.0.1:8081/predict"

BHK_MEAN = 2.083860
BHK_STD = 0.832256
Rent_MEAN = 3.499345e+04
Rent_STD = 7.810641e+04
Size_MEAN = 967.490729
Size_STD = 634.202328
Area_type = ["Carpet Area", "Super Area"]
City = ["Bangalore", "Chennai", "Delhi", "Hyberabad", "Kolkata", "Mumbai"]
Furnishing_status = ["Unfurnished", "Semi-furnished", "Furnished"]
Tenant_preferred = ["Bachelors", "Bachelors/Family", "Family"]
Bathroom_MEAN = 1.965866
Bathroom_STD = 0.884532
Total_floors_MEAN = 6.968816
Total_floor_STD = 9.467101
Point_of_Contact = ["Contact Owner", "Contact Agent"]



def generate_traffic():
    """
    Sends continuous requests to prediction API
    """

    headers = {"Content-Type": "application/json"}

    while True:
        total_floors = max(1, random.uniform(
            Total_floors_MEAN - 2 * Total_floor_STD,
            Total_floors_MEAN + 2 * Total_floor_STD
        ))

        test_data = {
            "BHK" : max(1, random.uniform(
                BHK_MEAN - 2 * BHK_STD, BHK_MEAN + 2 * BHK_STD
            )),
            "Rent" : max(1, random.uniform(
                Rent_MEAN - 2 * Rent_STD, Rent_MEAN + 2 * Rent_STD
            )),
            "Size" : max(1, random.uniform(
                Size_MEAN - 2 * Size_STD, Size_MEAN + 2 * Size_STD
            )),
            "Area Type" : random.choice(Area_type),
            "City" : random.choice(City),
            "Furnishing Status" : random.choice(Furnishing_status),
            "Tenant Preferred" : random.choice(Tenant_preferred),
            "Bathroom" :  max(1, random.uniform(
                Bathroom_MEAN - 2 * Bathroom_STD,
                Bathroom_MEAN + 2 * Bathroom_STD
            )),
            "Total Floors" :  total_floors,
            "Floor" : random.randint(-1, total_floors),
            "Point of Contact" : random.choice(Point_of_Contact)
        }
        payload = json.dumps(test_data)

        response = requests.request("POST", URL, headers=headers, data=payload)
        print(response.content)
        time.sleep(0.2)


if __name__ == "__main__":
    generate_traffic()