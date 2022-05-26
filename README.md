# Overview
Medical Chatbot made by -
Aaditya Singh
Prahlad Kumar
Rishi Srivastava
Saransh Gupta
Yash Karnawal

We have created Medchat: The Medical Chatbot as the final year project (NIT Jalandhar)

# Requirements

Python 3.5 or newer.


# Install requirements

Dependencies needed to be installed to run the project 

For Windows:

```
py -3 -m venv venv
venv\Scripts\activate
pip install flask torch nltk numpy==1.19.3 sklearn pandas matplotlib
```



For tokenization using nltk follwing is needed to be installed

```python
import nltk
nltk.download('punkt')
```

This will install all the required dependencies needed to run the application successfully.

## Run

To run MedicalChatbot, `cd` into MedicalChatbot repo on your computer and run `python -m flask run`. This will run the Flask 
server in development mode on localhost, port 5000.

`* Running on http://127.0.0.1:5000/ `
