FROM python:3.11-slim
WORKDIR /home
#COPY scripts/ /home/scripts/
#COPY examples/ /home/examples/
COPY ./requirements.txt /home
EXPOSE 8501
RUN pip install --no-cache-dir  --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
#RUN cd /home/ && echo $(ls -la)
ENTRYPOINT ["streamlit","run","/thesis_project/scripts/streamlit.py","--server.port=8501","--server.address=0.0.0.0","--server.runOnSave=true"]
