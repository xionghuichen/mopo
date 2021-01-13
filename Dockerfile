FROM sanluosizhou/selfdl:latest
RUN pip install git+https://github.com/xionghuichen/RLAssistant \
            gtimer \
            viskit \
            dataclasses \
            tabulate \
            ray[tune]==0.8.5 \
            serializable \
            dotmap
COPY environment /root/mopo_env
RUN pip install -r /root/mopo_env/requirements.txt