demo:
	python demo.py

build:
	sudo docker build -t ms-tf-seg .

run:
	sudo docker run --name ms-tf-seg -p 8080:80 ms-tf-seg