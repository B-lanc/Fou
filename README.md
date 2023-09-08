Building the docker image
> docker build . -t b-lanc/fou

Running the docker container
> docker run -dit --name=Fou_Dev1 --runtime=nvidia --gpus=0 --shm-size=2gb -v /mnt/Data/datasets/musdb18hq:/dataset -v /mnt/Data2/DockerVolumes/Fou_Dev1:/saves -v .:/workspace b-lanc/fou

Going into the docker container
> docker exec -it Fou_Dev1 /bin/bash

