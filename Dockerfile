# Use the official Python slim runtime as base image
FROM python:3.11-slim

# Set environment variables to keep python output clean and unbuffered
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Establish working directory inside the container
WORKDIR /app

# Install package dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK and TextBlob database corpora for NLP processing
RUN python -m nltk.downloader stopwords punkt wordnet
RUN python -c "from textblob import download_corpora; download_corpora.download_all()"

# Copy frontend templates and backend source files into Docker container
COPY frontend /frontend
COPY backend /app

# Expose port 7860 which Hugging Face Spaces requires
EXPOSE 7860

# Command to execute the Flask application
CMD ["python", "app.py"]
