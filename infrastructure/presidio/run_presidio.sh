
## For PII anonymization in text
## For PII detection and anonymization in text, the presidio-analyzer and presidio-anonymizer modules are required.

# Download Docker images
docker pull mcr.microsoft.com/presidio-analyzer
docker pull mcr.microsoft.com/presidio-anonymizer

# Run containers with default ports
docker run -d -p 5001:3000 mcr.microsoft.com/presidio-analyzer:latest

docker run -d -p 5002:3000 mcr.microsoft.com/presidio-anonymizer:latest

## For PII redaction in images
## For PII detection in images, the presidio-image-redactor is required.

# Download Docker image
docker pull mcr.microsoft.com/presidio-image-redactor

# Run container with the default port
docker run -d -p 5003:3000 mcr.microsoft.com/presidio-image-redactor:latest

## REST APIs
https://microsoft.github.io/presidio/api-docs/api-docs.html

