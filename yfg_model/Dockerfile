FROM amancevice/pandas:0.23.4-python3

RUN pip install --upgrade pip
RUN pip install synapseclient
RUN pip install git+https://github.com/Theano/Theano.git
RUN pip install git+https://github.com/Lasagne/Lasagne.git
RUN git clone https://github.com/philerooski/mHealthTools.git /root/mhealthtools
RUN cd /root/mhealthtools/ && git checkout reproduce_yfg_model && cd yfg_model

CMD predict.py
