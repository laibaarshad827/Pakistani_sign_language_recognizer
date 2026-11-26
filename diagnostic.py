import streamlit as st
import sys
from pathlib import Path

st.title("PSL APPLICATION DIAGNOSTICS")
st.write("1. Streamlit rendering works")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from database import Database
    st.success("2. Database import passed")

    from auth import Auth
    st.success("3. Authentication import passed")

    database = Database()
    st.success("4. Database initialization passed")

    auth = Auth(database)
    st.success("5. Authentication initialization passed")

    st.success("ALL BASIC CHECKS PASSED")

except Exception as error:
    st.exception(error)
