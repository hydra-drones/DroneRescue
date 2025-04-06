import streamlit as st

from v2_1.app_logic.utils.callbacks import CallbackResponse


def handle_callback_response(response: CallbackResponse):
    """Process the callback response stored in session state"""
    if response.success:
        st.success(response.message)
    else:
        if response.status_code == 400:
            st.warning(response.message)
        elif response.status_code >= 500:
            st.error(response.message)
        else:
            st.info(response.message)


def execute_callback(callback_func, *args):
    """Execute the callback function and handle its response"""
    response = callback_func(*args)
    handle_callback_response(response)
