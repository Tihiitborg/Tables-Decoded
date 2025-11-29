import streamlit as st
from PIL import Image
from tables.main import perform_td, perform_tsr, get_full_page_hocr
from tables.utils import draw_bboxes
import os
import pytesseract

# Title of the app
st.title("Table Reconstruction Tool")

# st.image("resources/iitb-bhashini-logo.png", use_column_width=True)

# 1. Image Uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

# 2. Dropdown for mode selection
mode = st.selectbox("Choose a mode:",
                    ["Table Detection",
                     "TSR - Struc ONLY",
                     "TSR - Struc + Content",
                     "Full Page Reconstruction"])

language = st.text_input(label= "Enter language here [Valid only for OCR]", value="eng")
langs = pytesseract.get_languages()
avail_langs = 'Available languages are : ' + str(langs)
st.text(avail_langs)
# Go Button
go = st.button("GO")

# Process the image and show output
if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    # Resizing images for good quality
    with open(os.path.join('uploads/', uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    img_path = os.path.join('uploads/', uploaded_file.name)
    print(img_path)
    print('IMAGE PATH ABOVE')
    image = Image.open(img_path)
    width, height = image.size
    if width < 2048:
        image = image.resize((width * 3, height * 3))
    image.save(img_path)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
else:
    st.write("Please upload an image to proceed.")

if uploaded_file is not None and go:
    # 3. Show output based on the mode
    if mode == "Table Detection":
        print(img_path)
        result = perform_td(img_path)
        print(result)
        processed_image = draw_bboxes(img_path, result, (255, 0, 128), 2)
        st.image(processed_image, caption='Table Detection Output', use_column_width=True)

    elif mode == "TSR - Struc ONLY":
        # Placeholder logic for TSR - Struc ONLY: returns HTML-like table structure
        html_output, struct_cells = perform_tsr(img_path, 0, 0, True, language)
        processed_image = draw_bboxes(img_path, struct_cells, (0, 128, 255), 1)
        st.image(processed_image, caption = 'TSR Output', use_column_width = True)
        print(html_output)
        st.markdown(html_output, unsafe_allow_html = True)

    elif mode == "TSR - Struc + Content":
        # Placeholder logic for TSR - Struc + Content: returns HTML-like structure and content
        html_output, struct_cells = perform_tsr(img_path, 0, 0, False, language)
        processed_image = draw_bboxes(img_path, struct_cells, (0, 128, 255), 1)
        st.image(processed_image, caption = 'TSR Output', use_column_width = True)
        print(html_output)
        st.markdown(html_output, unsafe_allow_html = True)

    elif mode == "Full Page Reconstruction":
        # Placeholder logic for Full Page Reconstruction: returns full page HTML-like structure
        html_output = get_full_page_hocr(img_path, language)
        st.markdown(html_output, unsafe_allow_html=True)
        st.write("Full page reconstruction generated.")
