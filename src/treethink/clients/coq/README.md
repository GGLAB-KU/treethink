# Rocq Language Support via rocq-ml-toolbox

Rocq language is supported via the [rocq-ml-toolbox](https://github.com/LLM4Rocq/rocq-ml-toolbox/tree/main) project, which provides a set of tools and libraries for working with the Rocq language. 

Installing the server and running it is the same as specified in the repo's README file, giving it again here for convenience:

For docker:
```bash
docker run --rm -p 5000:5000 theostos/coq-mathcomp:9.0-2.5.0 rocq-ml-server --host 0.0.0.0 \
--port 5000 \
--num-pet-server 2 \
--workers 3
```

For local installation:
```bash
rocq-ml-server --num-pet-server 4 --workers 9 --port 5000
```