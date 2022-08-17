# 
FROM python:3.10

# 
WORKDIR /code
EXPOSE 8000
ENV PORT 8000
ENV HOST 0.0.0.0

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./app /code/app

# 
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]