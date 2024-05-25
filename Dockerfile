FROM python:3.8

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "--server.port", "8501", "app.py"]
