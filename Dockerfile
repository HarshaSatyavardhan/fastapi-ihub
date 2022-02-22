FROM tiangolo/uvicorn-gunicorn:python3.8-slim
COPY ./requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir --upgrade -r ./requirements.txt
COPY . ./

RUN chmod +x entrypoint.sh
CMD [ "timeout", "600","./entrypoint.sh" ]