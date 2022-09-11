import os
import time
import requests
from pymongo import MongoClient
import streamlit as st

PREDICTION_SERVICE_HOST = os.getenv("PREDICTION_SERVICE_HOST")
PREDICTION_SERVICE_PORT = os.getenv("PREDICTION_SERVICE_PORT", "8081")
MONGODB_USERNAME = os.getenv("MONGODB_USERNAME")
MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD")
MONGODB_CLUSTER_URL = os.getenv("MONGODB_CLUSTER_URL")
SUGGESTIONS_DATABASE_NAME = os.getenv("SUGGESTIONS_DATABASE_NAME", "suggestions")

api_endpoint_url = f"{PREDICTION_SERVICE_HOST}:{PREDICTION_SERVICE_PORT}/predict"

mongodb_uri = f"mongodb://{MONGODB_USERNAME}:{MONGODB_PASSWORD}@{MONGODB_CLUSTER_URL}/?retryWrites=true&w=majority"
mongodb_client = MongoClient(mongodb_uri)
db = mongodb_client.get_database(f"{SUGGESTIONS_DATABASE_NAME}")
collection = db.get_collection(f"{SUGGESTIONS_DATABASE_NAME}")

st.markdown(
    "<h1 style='text-align: center; color: red;'>Welcome to your House Rent Prediction App</h1>",
    unsafe_allow_html=True
)

image_url = "https://my-esl-bucket.s3.ap-south-1.amazonaws.com/image.jpg"
st.image(image_url, width=700)

st.markdown(
    "<h4 style='text-align: center; color: blue;'>Predict the rent of your future house in India !!!</h4>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style='text-align: center;'> It's very simple to use. 
    All you have to do is fill ... the form below with the desired characteristic of your future house 
    and we'll give you an idea of the price for renting a house with your characteristics. </p>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown(
        "<h2 style='text-align: center;'>Context</h4>",
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <p style='text-align: justify;'>
        Housing in India varies from palaces of erstwhile maharajas to modern apartment buildings in big cities to tiny huts in far-flung villages. 
        There has been tremendous growth in India's housing sector as incomes have risen. The Human Rights Measurement Initiative finds that India is doing 60.9% of 
        what should be possible at its level of income for the right to housing.
        Renting, also known as hiring or letting, is an agreement where a payment is made for the temporary use of a good, service, 
        or property owned by another. A gross lease is when the tenant pays a flat rental amount and the landlord pays 
        for all property charges regularly incurred by the ownership. Renting can be an example of the sharing economy. </p>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        "<h2 style='text-align: center;'>About</h4>",
        unsafe_allow_html=True
    )
    st.markdown(
        """<p style='text-align: justify'> This App is designed to help you get an idea of the price for renting your future house in India. 
        We trained our machine learning model with an open-source dataset from <a href='https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset'>Kaggle</a>.
        This dataset contains more than 4000 houses rented. As soon as we'll get new data, we will retrain our model to improve his performance so that 
        you will get the most accurate prediction. </p>""",
        unsafe_allow_html=True
    )


area_types = ("Carpet Area", "Super Area")
cities = ("Bangalore", "Chennai", "Delhi", "Hyberabad", "Kolkata", "Mumbai")
furnishing_status_list = ("Unfurnished", "Semi-furnished", "Furnished")
tenant_preferred = ("Bachelors", "Bachelors/Family", "Family")
point_of_contact = ("Contact Owner", "Contact Agent")

st.markdown("""---""")
bhk = st.number_input(
    label = "Enter the number of BHK (Bedrooms, Hall, Kitchen).",
    min_value = 1,
    max_value = 10
)
bathroom = st.number_input(
    label = "Enter the number of Bathroom(s).",
    min_value = 1,
    max_value = 10
)
area_type = st.selectbox(
    label = "What is the Area Type ?",
    options = area_types
)
size = st.number_input(
    label = "What is the size of the house in Square Feet ?",
    min_value = 9,
    max_value = 9000
)
total_floors = st.number_input(
    label = "What is total number of floors in which your house will be ?",
    min_value = 1,
    max_value = 100,
)
floor = st.number_input(
    label = f"""In which floors would you like your house to be 
    (-1 for 'lower basement', 0 for 'ground') """,
    min_value = -1,
    max_value = int(total_floors)
)
city = st.selectbox(
    label = "In which city do you want to rent this house ?",
    options = cities
)
furnishing_status = st.selectbox(
    label = "Select the furnishing status of the house.",
    options = furnishing_status_list
)
contact = st.selectbox(
    label = "Whom would you like to contact for more information regarding this house ?",
    options = point_of_contact
)
tenant_preferred = st.selectbox(
    label = "Select the Tenant Preferred  by the Owner or Agent.",
    options = tenant_preferred
)
    
predict_button = st.button("Predict")
st.markdown("""---""")
    
suggestions = st.checkbox('Check this box if you have any suggestions you would like to tell us.')

if suggestions:
    with st.form("suggestion-form"):
        suggestions = st.text_area(
            label='Please, write your suggestions here',
            value = "",
        )
        submit = st.form_submit_button("Submit")
        if submit:
            suggestions = {
                "suggestions" : suggestions
            }
            collection.insert_one(suggestions)
            st.success("Thank you for your suggestions, we'll try to take into account. Good Bye 😎")


def predict(bhk, bathroom, size, area_type, total_floors, floor, city, furnishing_status, contact, tenant_preferred):
    """
    A function that sends a prediction request to the API and return the predicted price.

    Args:
        - bhk : 1st input
        - bathroom : 2nd input
        - area_type : 3rd input
        - size : 4th input
        - total_floors : 5th input
        - floor : 6th input
        - city : 7th input
        - furnishing_status : 8th input
        - contact : 9th input
        - tenant_prefered : 10th input
    """
    # Convert the bytes image to a NumPy array
    inputs = {
        'BHK': bhk,
        'Bathroom' : bathroom,
        'Size' : size,
        'Area Type' : area_type,
        'Total Floors' : total_floors,
        'Floor' : floor,
        'City' : city,
        'Furnishing Status' : furnishing_status,
        'Tenant Preferred' : tenant_preferred,
        'Point of Contact' : contact
    }

    # Send the inputs to the API
    response = requests.post(api_endpoint_url, json=inputs).json()
    print(inputs)

    if response.status_code == 200:  # pylint: disable=no-else-return
        return response['house_rent_prediction']
    else:
        raise Exception(f"Status: {response.status_code}")


# This function returns a success answer if everything goes right
def main():
    if predict_button:
        with st.spinner("Predicting..."):
            time.sleep(2)
            try:
                prediction = predict(
                    bhk, bathroom, size, area_type, total_floors, floor, city, furnishing_status, contact, tenant_preferred
                )
                st.success(f"The predicted price of house for renting is {prediction} 😎")
            except Exception:  # pylint: disable=broad-except
                st.warning('Something went wrong ... please check your inputs and try again 😳')

if __name__ == "__main__":
    main()