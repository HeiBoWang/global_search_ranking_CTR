FROM registry.cn-hangzhou.aliyuncs.com/dazhuanjia/search-rank-api:python3.6
COPY . .
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
CMD ["gunicorn", "search_rank_ctr:app","-c","./gunicorn.conf.py"]



