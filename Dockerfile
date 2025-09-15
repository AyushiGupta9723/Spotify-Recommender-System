# set up the base image
FROM python:3.12-slim

# set the working directory
WORKDIR /app/

# copy the requirements file to workdir
COPY requirements.txt .

# install the requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy all required data files at once

COPY src/ ./src/
COPY data/ ./data/


# Copy all required Python scripts at once
COPY app.py .

# expose the port on the container
EXPOSE 8501

# run the streamlit app
CMD ["streamlit", "run", "app.py"]