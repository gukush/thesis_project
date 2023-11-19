## Pipeline for data extraction and summarization from business reports
This repository contains code for thesis project. The necessary environment is set up via docker. You can build it by calling following command while inside repository
```
docker build -t thesis_project .
```
In order to start the container please run from within the repository:
```
docker run -p 8501:8501 \
-it \
--name thesis_project \
--mount type=bind,source="$(pwd)",target=/thesis_project \
thesis_project
```
The "--mount" option is responsible for sharing directories between docker container and the host system, so that changes made to this repository locally will be updated inside the container too. Instead of the "$(pwd)" part one can insert the path to this repository.

After running on address http://0.0.0.0:8501/ ( http://localhost:8501 on Windows ) there will simple page created in streamlit to help prototype the solution. If not already enabled, please check the "Run on Save" option in page's setting to allow changes to instantly update after saves.

In order to launch shell to run python manually please input following command (while container is running):
```
docker exec -it thesis_project bash
```
## Warning!
Whole docker container's size is more than 5.7 GB due to using torch package as well as AI models.
