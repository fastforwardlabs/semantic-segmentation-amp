# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2022
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

import os
import streamlit as st
import matplotlib.pyplot as plt

from apps.app_utils import (
    get_dataset,
    get_data_pipeline,
    get_model,
    build_samples_queue,
    get_next_example,
    prepare_visual_assets,
    toggle_show_masks,
    save_channel_imgs,
    save_mask_overlay_image,
    CLASS_MAP,
)

# ------------------------- PAGE SETUP -------------------------
ffl_favicon = plt.imread("images/cldr-favicon.ico")
st.set_page_config(
    page_title="Manufacturing Defect Detection",
    page_icon=ffl_favicon,
    layout="wide",
)
sd = get_dataset()
sdp = get_data_pipeline()
unet_model = get_model()

# ------------------------- INITIALIZE SESSIONS STATE -------------------------
# set session state variable to hold queue of images
if "samples_dict" not in st.session_state:
    st.session_state["samples_dict"] = build_samples_queue(sd, n_samples=10)

# set session state variable to track current index for each defect type
if "class_index_tracker" not in st.session_state:
    st.session_state["class_index_tracker"] = {v: -1 for v in CLASS_MAP.values()}

# set session state variable for mask toggle
if "show_masks" not in st.session_state:
    st.session_state["show_masks"] = False

# set session state variable for defect type selection
if "defect_type" not in st.session_state:
    st.session_state["defect_type"] = list(CLASS_MAP.keys())[0]

# initialize selected example from queue
if "current_x" not in st.session_state.keys():
    get_next_example()

st.title(":mag: Manufacturing Defect Detection")

st.write(
    "The computer vision task of _semantic segmentation_ involves predicting the label of each pixel in an image with a corresponding \
    class of what that pixel represents. This application is intended to help visualize the quality of segmentation mask predictions on the \
    [Severstal Steel Defect Detection](https://www.kaggle.com/competitions/severstal-steel-defect-detection/overview) dataset - a dataset of \
    steel sample images with several types of surface defects. Analyzing various defects helps build intuition around a segmentation models' strengths and shortcomings."
)
st.write(
    "To get started, select the defect type you'd like to visualize (on the left). Then, click 'Predict Segmentation Masks' to generate mask predictions.\
        The mask predictions can be visualized as individual channels (one for each defect class) or overlayed on the actual image together. In either case, \
        we provide a side-by-side comparison with the ground truth segmentation masks for quick assessment of quality."
)
st.write("")

# ------------------------- SIDEBAR -------------------------
ffl_logo = plt.imread("images/ffllogo2@1x.png")
st.sidebar.image(ffl_logo)

with st.sidebar:
    st.sidebar.markdown("## Select a defect type")
    defect_type_selector = st.selectbox(
        "What defect type would you like to work with?",
        CLASS_MAP.keys(),
    )

    if st.session_state["defect_type"] != defect_type_selector:
        # if defect type is different than stored session state,
        # update and get a new example
        st.session_state["defect_type"] = defect_type_selector
        get_next_example()

    st.sidebar.markdown("## Try another image")
    st.sidebar.caption(
        "Click the button below if you wish to visualize segmentation mask predictions for a different image of the selected defect type."
    )
    next_example = st.button(
        "Next example",
        on_click=get_next_example,
    )

# ------------------------- MAIN CONTENT -------------------------

x, y_true, y_pred = prepare_visual_assets(
    x=st.session_state["current_x"],
    y=st.session_state["current_y"],
    pipeline=sdp,
    model=unet_model,
)

st.image(
    image=x,
    caption=f"Example Image with Defect Type: {defect_type_selector}",
    use_column_width=True,
)

col_1, col_2 = st.columns([3, 1])
with col_2:
    placeholder = st.empty()
    predict_button = placeholder.button(
        "Predict Segmentation Masks", disabled=False, on_click=toggle_show_masks, key=1
    )
    if predict_button:
        placeholder.button("Predict Segmentation Masks", disabled=True, key=2)


if st.session_state["show_masks"]:
    tab1, tab2 = st.tabs(["Mask Overlay", "Separate Channels"])

    with tab1:
        save_mask_overlay_image(x, y_true, type="true")
        save_mask_overlay_image(x, y_pred, type="pred")

        st.write("**Predicted Mask**")
        st.image(
            "apps/tmp/pred/overlay.png",
            clamp=True,
            caption=f"Predicted Overlay",
            use_column_width=True,
        )

        st.write("**Ground Truth Mask**")
        st.image(
            "apps/tmp/true/overlay.png",
            clamp=True,
            caption=f"Ground Truth Overlay",
            use_column_width=True,
        )

    with tab2:
        save_channel_imgs(y_pred, type="pred")
        save_channel_imgs(y_true, type="true")

        sbs_col1, sbs_col2 = st.columns(2)

        with sbs_col1:
            st.write("**Predicted Mask**")
            st.image(
                x,
                clamp=True,
                caption=f"Model Input",
                use_column_width=True,
            )
            st.image(
                "apps/tmp/pred/channel_1.png",
                clamp=True,
                caption=f"Scratches Channel",
                use_column_width=True,
            )
            st.image(
                "apps/tmp/pred/channel_2.png",
                clamp=True,
                caption=f"Patches Channel",
                use_column_width=True,
            )

        with sbs_col2:
            st.write("**Ground Truth Mask**")
            st.image(
                x,
                clamp=True,
                caption=f"Model Input",
                use_column_width=True,
            )
            st.image(
                "apps/tmp/true/channel_1.png",
                clamp=True,
                caption=f"Scratches Channel",
                use_column_width=True,
            )
            st.image(
                "apps/tmp/true/channel_2.png",
                clamp=True,
                caption=f"Patches Channel",
                use_column_width=True,
            )
