# Scoreboard - Image to Excel table

This project uses CNN, Tesseract, Houghline, etc to convert (python)

## Getting Started

```
git clone https://github.com/fxanhkhoa/IPS.git
```
```
git checkout IPS-Score-board_CNN-model
```

### Prerequisites

Install [PyQt5](https://pypi.org/project/PyQt5/)

```
pip3 install PyQt5
```

Install [Openpyxl](https://pypi.org/project/openpyxl/)

```
pip3 install openpyxl
```

Install [Tensorflow](https://pypi.org/project/tensorflow/)

```
pip3 install tensorflow
```

Install [Keras](https://pypi.org/project/Keras/)

```
pip3 install Keras
```

Install [Pytesseract](https://www.miai.vn/2019/08/22/ocr-nhan-dang-van-ban-tieng-viet-voi-tesseract-ocr/)

## Running the tests

### With realize project:

```
cd Segment
cd UI
python3 mainRun.py
```
* Step 1: Choose Image
  + In "Filedialog", choose the path of the image that you need to convert to Excel table (*.JPG)
  + In "Select 4 corners of The Scoreboard", click on 4 corners of the Scoreboard table

* Step 2: Start the process
  + Click "Start" and wait a secs ( Just wait .-. )
  + In "Handwriting columns (press [q] to next)",  Click at the top of the column that have the number of Handwriting

* Step 3: Save the Excel table
  + Click "Save" and choose where u want to save
  + Get the name and save it

### With dev - test project:

Traing: Just do anything u need to understand it .-. 

```
cd Model
```
Segment code: 

```
cd Segment
cd "Raw code"
```

```
python3 transform_example.py -i "path-image"
python3 Text-Segmentation.py -i Warped.jpg
```

## Authors

* **Ly Hong Phong** - *Develope and Operation* - [phonghongs](https://github.com/phonghongs)

* **Che Quang Huy** - *Develope and Operation* - [chequanghuy](https://github.com/chequanghuy)

## License

