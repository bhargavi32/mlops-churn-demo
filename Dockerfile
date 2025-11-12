# Step 1: Use a lightweight Python image
FROM python:3.10

# Step 2: Set working directory inside the container
WORKDIR /app

# Step 3: Copy everything from your local folder to the container
COPY . .

# Step 4: Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Expose port 8080
EXPOSE 8080

# Step 6: Run FastAPI app using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
