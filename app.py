import streamlit as st

from recommender import get_rekomendasi


st.title("Dessert Recommendation System")

menu = st.selectbox(
    "Pilih Menu",
    ["donut", "cheesecake", "ice_cream"]
)

flavor = st.selectbox(
    "Pilih Flavor",
    ["matcha", "vanilla", "chocolate"]
)

price = st.selectbox(
    "Pilih Budget",
    [
        "Rp 1-25.000",
        "Rp 25.000-50.000",
        "Rp 50.000-100.000"
    ]
)

dine = st.selectbox(
    "Dine Option",
    ["dine_in", "takeaway", "both"]
)

rating = st.slider(
    "Minimum Rating",
    1.0,
    5.0,
    4.0
)

if st.button("Cari Rekomendasi"):

    result = get_rekomendasi(
        menu=menu,
        flavor=flavor,
        price=price,
        dine=dine,
        rating=rating
    )

    st.dataframe(result)
