FROM python:3.12-slim

# Install libGL and other required packages
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Set work directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set default command
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
