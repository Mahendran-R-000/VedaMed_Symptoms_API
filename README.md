

# VedaMed Symptom API

VedaMed Symptom API is a Flask application that utilizes machine learning models to predict diseases based on patient symptoms. This backend application provides API endpoints for disease classification and symptom synonym generation.

## Getting Started

To get started with the VedaMed Symptom API, follow these steps:

### Prerequisites

Before running the API, ensure you have the following installed:

- Python 3.x
- Required Python packages (install using `pip install -r requirements.txt`)
- NLTK data (download using `nltk.download('wordnet')` and `nltk.download('stopwords')`)

### Installation

1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Install the required Python packages by running:

   ```
   pip install -r requirements.txt
   ```

4. Download NLTK data by running:

   ```
   python -m nltk.downloader wordnet
   python -m nltk.downloader stopwords
   ```

## API Endpoints

### Disease Classification

- **Endpoint:** /disease
- **Method:** POST
- **Parameters:** 
  - symptoms (comma-separated list of symptoms)
- **Response:** JSON object containing predicted diseases based on the provided symptoms.

### Symptom Synonym Generation

- **Endpoint:** /EnterSymptoms
- **Method:** POST
- **Parameters:** 
  - user_symptoms (comma-separated list of symptoms)
- **Response:** JSON object containing synonyms of the provided symptoms.

## Running the Application

To run the VedaMed Symptom API locally, execute the following command in your terminal:

```
python main.py
```

The API will start running on `http://127.0.0.1:5000/`.

## Contributing

Contributions to the VedaMed Symptom API are welcome! If you encounter any bugs or have feature requests, please open an issue or submit a pull request on GitHub.

Feel free to customize this README file further according to your project's specific requirements. Let me know if you need any further assistance!
