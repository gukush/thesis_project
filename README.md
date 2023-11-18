### Pipeline for data extraction and summarization from business reports
This repository contains code for thesis project. The necessary environment is set up via docker. You can build it by calling
```
docker build -t thesis_project .
```
In order to start the container please run
```
docker run -p 8501:8501 -it --name thesis_project thesis_project
```
In order to launch shell to run python manually please input following command (while container is running):
```
docker exec -it 
```
## Warning!
Whole docker container's size is more than 5.7 GB due to using torch package as well as AI models.
