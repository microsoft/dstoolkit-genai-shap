FROM python:3.9
WORKDIR /workspace

EXPOSE 8888

CMD ["make", "run-in-container"]
