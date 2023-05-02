# Chat With ~~IGXLHelp~~Your PDF
This is an application based on Azure OpenAI.  
This project demonstrates a web application that allows users to chat with a PDF file using natural language. The app consists of an HTML interface and a Flask backend.  

### **NOTE: Azure OpenAI API is needed!!!**

## Features

- Users can ask questions related to a specific PDF file uploaded

## Project Structure

The project consists of two main files:

1. `index.html`: The main HTML file that contains the frontend interface for the user. It includes:
    - A text input for users to ask questions
    - A submit button to send the question to the backend
    - An area to display AI-generated responses
    - An upload button to upload a PDF file
    - A submit button to save the uploaded PDF file

2. `server.py`: The Flask backend that handles user input and file uploads. It includes:
    - A route for the main page, which renders the `index.html` file
    - A route for file uploads, which saves the uploaded PDF file to a specified folder
    - A route for processing user questions and returning AI-generated responses

3. 'src': Include all the bakend code to setup Chatbot.

## Usage

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```
2. Prepare a "key.txt" file with your Azure OpenAI key, and put it in the same path as server.py.

3. Run the Flask application:

```bash
python server.py
```

4. Open your browser and navigate to http://localhost:3000 to access the web application.
5. Upload a PDF file by clicking the "Upload PDF" button and selecting a file. 
6. Click the "Upload" button to save the uploaded PDF file to the server.
7. Enter a question related to a PDF file and click the "Submit" button. 