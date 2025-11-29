# table-extractor
Consists of various table-related inference calls for table reconstruction in documents. 
All the code is encapsulated in the 'tables' directory.
The 'uploads' directory has sample images.


## Setting Up

#### Install the required dependencies
```commandline
pip install -r requirements.in
```
#### Download the model
Download sprint.pt from the Releases Section and place it in 'tables/model' directory.


## Source Code Details
Following table calls are integrated in this repository

### table-detection
Based on our trained Yolo model equipped for multilingual table detection. 

```commandline
python3 infer.py <page-image-path> td True
```

### table-structure-recognition
Based on SPRINT, our script-agnostic table structure recognizer can predict OTSL sequences.

```commandline
python3 infer.py <table-image-path> tsr True
```

### full-page-reconstrcution
Uses YOLO-based table detector, SPRINT and Tesseract to generate an HOCR composed of text and tables in the inoput page image.

```commandline
python3 infer.py <page-image-path> ocr True
```


## Containerization

### Building Image
```
cd tables
docker build -t tablecalls .
```

### Running Container
```
docker run --rm --gpus all -it -v '/data/DHRUV/Document-OCR-App/document-layout-ocr/uploads/table.jpg':/docker/uploads/tables.jpg tablecalls uploads/table.jpg tsr False
```

## User Interface
Uses streamlit to run all the required calls

```commandline
streamlit run api.py
```
