## Installation

There are 2 ways to install PreyTouch
1. Using docker (easy, but you must have docker and docker-compose installed)
2. Manually (hard)

### Installing using docker
Get into the PreyTouch directory, and then cd into the docker folder
```console
cd docker/
```
create .env file, and insert the following
```dotenv
ROOT_DIR=<path_to_PreyTouch_dir>
```
```console
docker-compose up -d
```
### Installing manually