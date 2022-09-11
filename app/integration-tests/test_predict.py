#!/usr/bin/env python
# coding: utf-8

"""Integration tests module"""
import json
import requests
from deepdiff import DeepDiff

PREDICT_URL = 'http://127.0.0.1:8081/predict'
HEADER = {'Content-Type': 'application/json'}

inputs = {
    'BHK': 3,
    'Size': 1000,
    'Area Type': 'Carpet Area',
    'City': 'Mumbai',
    'Furnishing Status': 'Furnished',
    'Tenant Preferred': 'Bachelors/Family',
    'Bathroom': 2,
    'Floor': 5,
    'total_floors': 15,
    'Point of Contact': 'Contact Agent',
}

actual_response = requests.post(
    PREDICT_URL,
    data=json.dumps(inputs),
    headers=HEADER,
).json()

expected_response = {"house_rent_prediction": 98352}

diff = DeepDiff(actual_response, expected_response, significant_digits=1)
print(f'diff={diff}')

assert 'type_changes' not in diff
assert 'values_changed' not in diff
