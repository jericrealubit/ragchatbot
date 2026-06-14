# Use an official lightweight Python image matching your project
FROM python:3.11-slim

# Set up a new user named "user" for security (Hugging Face requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory inside the container
WORKDIR $HOME/app

# Copy your local requirements file and install dependencies
COPY --chown=user requirements.txt $HOME/app/
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of your application code into the container
COPY --chown=user . $HOME/app

# Chainlit runs on port 7860 by default on Hugging Face Spaces
EXPOSE 7860

# The command to start your Chainlit app
CMD ["chainlit", "run", "solutions/langchain/chatpdf.py", "--host", "0.0.0.0", "--port", "7860", "-h"]