build-image:
	docker build -t genaishap .

run-container:
	docker run --rm -it -p 8888:8888 -v $$(pwd):/workspace genaishap

run-in-container:
	pip install --no-cache-dir -r requirements.txt
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
