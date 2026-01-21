qdrant_dir=`find ${HOME}/workspace -name "docker-compose.yml" 2> /dev/null | grep qdrant | xargs dirname`


qdrant_build:
	make qdrant_down && cd ${qdrant_dir} && docker-compose build

qdrant_down:
	cd ${qdrant_dir} && docker-compose down --volumes

qdrant_run:
	make qdrant_down && cd ${qdrant_dir} && docker-compose up -d

qdrant_stop:
	cd ${qdrant_dir} && docker-compose stop

qdrant_load:
	./load_qdrant.sh

qdrant_count:
	./qdrant_point_count.sh

hybrid_load:
	python hybrid_load_qdrant.py

hybrid_evaluate:
	python evaluate_hybrid_search.py

ibm_granite_run:
	./granite-4.0-micro.sh

ibm_granite_stop:
	pid=`ps -ef|grep llama_cpp.server|grep granite|grep -v grep|awk '{print $$2}'` && [ -n $${pid} ] && kill -9 $${pid}

run:
	~/workspace/simple_rag/.env . && nohup chainlit run simple_rag.py --port 8001 2>&1 > logs/chainlit.log &

stop:
	pid=`ps -ef|grep chainlit|grep python|grep -v grep|awk '{print $$2}'` && [ -n $${pid} ] && kill -9 $${pid}
