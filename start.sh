docker build -t model:latest .
docker run -d --name model -p 5000:5000 model

