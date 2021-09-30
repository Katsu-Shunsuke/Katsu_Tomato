## Docker Compose
This compose file spawns 5 containers (for now):
1. ros-master
2. dummy_camera (working on the ritsumeikan server at the moment so cannot connect zed mini cameras)
3. stereo_matching
4. instance_segmentation
5. synthesis

The following are the requirements:
- nvidia-docker 2.0.3
- docker 18.09.2
- nvidia-driver 430.64
- docker-compose 1.29.2

Note that individual images need to be built before running
```bash
$ docker-compose up
```
In this case I have named it shiotani_stereo which is what appears in docker-compose.yml
This can be done by executing the follwing inside the directory with Dockerfile:
```bash
$ docker build -t <container_name> .
```

