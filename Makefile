help:
	@cat Makefile

DATA?="/datasets/rpadilha"
HOME?="/home/rpadilha"
SRC="/work"#"/work/rpadilha"#"${HOME}/remote_works/dl-04"#$(shell dirname `pwd`)
GPU?=0
DOCKER_FILE=Dockerfile
DOCKER= nvidia-docker #GPU=$(GPU) nvidia-docker
TEST=tests/

build:
	docker build -t tensorflow_rpadilha -f $(DOCKER_FILE) .

bash: build
	$(DOCKER) run -it -v $(DATA):/datasets/rpadilha -v $(SRC):/work -e OUTSIDE_USER=rpadilha -e OUTSIDE_UID=10037 -e OUTSIDE_GROUP=`/usr/bin/id -ng rpadilha` -e OUTSIDE_GID=`/usr/bin/id -g rpadilha` --name rpadilha_tensorflow tensorflow_rpadilha bash

ipython: build
	$(DOCKER) run -it -v $(SRC):/src -v $(DATA):/data tensorflow ipython

notebook: build
	$(DOCKER) run -it -v $(SRC):/src -v $(DATA):/data --net=host tensorflow

test: build
	$(DOCKER) run -it -v $(SRC):/src -v $(DATA):/data tensorflow py.test $(TEST)

