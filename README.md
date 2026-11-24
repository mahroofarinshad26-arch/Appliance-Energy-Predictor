# Appliance Energy Predictor 

A Machine Learning-based Streamlit web application that predicts appliance energy consumption based on user-provided input parameters.

##  Project Overview

The **Appliance Energy Predictor** uses a trained Machine Learning pipeline to estimate appliance energy consumption.

The trained model is integrated into a **Streamlit** application, providing a simple and interactive interface for users to enter values and obtain predictions.

##  Features

* Interactive Streamlit web interface
* Machine Learning-based energy prediction
* Pre-trained ML pipeline using `joblib`
* Simple and user-friendly input interface
* Real-time prediction
* Web-based deployment using Streamlit

##  Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Git & GitHub

##  Project Structure

```text
Appliance-Energy-Predictor/
│
├── appliance_app.py
├── appliance_energy_pipeline.pkl
├── requirements.txt
├── .gitattributes
└── README.md
```

##  Requirements

The project uses the following Python libraries:

```text
streamlit
pandas
numpy
scikit-learn
```

##  Run the Project Locally

Clone the repository:

```bash
git clone https://github.com/mahroofarinshad26-arch/Appliance-Energy-Predictor.git
```

Navigate to the project directory:

```bash
cd Appliance-Energy-Predictor
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run appliance_app.py
```

The application will open in your web browser.

##  Deployment

This application can be deployed using **Streamlit Community Cloud** by connecting the GitHub repository and selecting:

```text
Branch: main
Main file: appliance_app.py
```

##  Machine Learning Model

The trained Machine Learning pipeline is stored in:

```text
appliance_energy_pipeline.pkl
```

The application loads this trained pipeline and uses it to generate energy consumption predictions based on the user's inputs.

##  Purpose

This project demonstrates how a Machine Learning model can be integrated into an interactive web application and deployed for practical use.


**Mahroofa Rinshad**

---

⭐ If you find this project useful, consider giving the repository a star!
