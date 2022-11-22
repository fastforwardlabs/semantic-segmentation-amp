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
    show_masks,
    save_channel_imgs,
    save_mask_overlay_image,
    CLASS_MAP,
)

# ------------------------- PAGE SETUP -------------------------
ffl_favicon = plt.imread("images/cldr-favicon.ico")
st.set_page_config(
    page_title="Severstal Steel Defect Detection",
    page_icon=ffl_favicon,
    layout="wide",
)
sd = get_dataset()
sdp = get_data_pipeline()
unet_model = get_model()

# ------------------------- INITIALIZE SESSIONS STATE -------------------------
# set session state variable to hold queue of images
if "samples_dict" not in st.session_state:
    st.session_state["samples_dict"] = build_samples_queue(sd, n_samples=5)

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
    show_masks()  # this only gets run here because we're intializing


st.title(":mag: Severstal Steel Defect Detection")
st.write(
    "This application is intended to help visualize the quality of segmentation mask predictions on the _Severstal Steel \
        Defect Detection_ dataset. Analyzing various defects helps build intuition around a given models strengths and shortcomings."
)
st.write(
    "To get started, select the defect type you'd like to visualize. Then, click 'Predict Segmentation Masks' to generate mask predictions.\
        These can be visualized as individual channels (one for each defect class) or overlayed on the actual image together. In either case, \
        we provide a side-by-side comparison with the ground truth segmentation masks for quick assessment of quality."
)

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

    st.sidebar.markdown("## Next example")
    st.sidebar.caption(
        "This application is intended to be run sequentially from top to bottom. If you wish to visualize segmentation mask \
        predictions for a different image or defect type, click the button below."
    )
    next_example = st.button(
        "Try another image",
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
        "Predict Segmentation Masks", disabled=False, on_click=show_masks, key=1
    )
    if predict_button:
        placeholder.button("Predict Segmentation Masks", disabled=True, key=2)


if st.session_state["show_masks"]:
    tab1, tab2 = st.tabs(["Mask Overlay", "Separate Channels"])

    with tab1:
        save_mask_overlay_image(x, y_true, type="true")
        save_mask_overlay_image(x, y_pred, type="pred")

        st.write("**Ground Truth Mask**")
        st.image(
            "apps/tmp/true/overlay.png",
            clamp=True,
            caption=f"Ground Truth Overlay",
            use_column_width=True,
        )

        st.write("**Predicted Mask**")
        st.image(
            "apps/tmp/pred/overlay.png",
            clamp=True,
            caption=f"Predicted Overlay",
            use_column_width=True,
        )

    with tab2:
        save_channel_imgs(y_pred, type="pred")
        save_channel_imgs(y_true, type="true")

        sbs_col1, sbs_col2 = st.columns(2)

        with sbs_col1:
            st.write("**Ground Truth Mask**")
            st.image(
                x,
                clamp=True,
                caption=f"Model Input",
                use_column_width=True,
            )
            st.image(
                "apps/tmp/true/channel_0.png",
                clamp=True,
                caption=f"Background Channel",
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

        with sbs_col2:
            st.write("**Predicted Mask**")
            st.image(
                x,
                clamp=True,
                caption=f"Model Input",
                use_column_width=True,
            )
            st.image(
                "apps/tmp/pred/channel_0.png",
                clamp=True,
                caption=f"Background Channel",
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
