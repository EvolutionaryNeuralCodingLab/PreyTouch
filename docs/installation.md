# Installation

## Install system dependencies

Install the apt-get dependencies before running the system. (This installation needs sudo permissions)

Get into the PreyTouch directory

```console
chmod +x Arena/scripts/arena_init.sh
./Arena/scripts/arena_init.sh
```

## Install dependencies using docker

PreyTouch has several dependencies that must be installed.

1. Redis
2. Bugs Application (Node.js)
3. Periphery service (communication with the Arduino)
4. PostgreSQL Database
5. Mosquitto MQTT

A manual for installing docker + docker-compose on ubuntu 22.04: <https://linuxhint.com/install-docker-compose-ubuntu-22-04>

```console
cd docker/
docker-compose up -d
cd ..
```

## Install pip packages

```console
pip install -r requirements/arena.txt
```

## Install nvidia-driver

Check if you have a nvidia-driver installed on your system, by running:

```console
nvidia-smi
```

If the above command is able to run, your'e all set and can skip this step, otherwise:

```console
sudo apt install nvidia-driver-525
```

PreyTouch only tested with this driver, but other nvidia-drivers should work as well. Notice you'll have to reboot after the driver installation.

## check PreyTouch

Now the preyTouch API should be able to run, check it by running:

```console
cd Arena
python api.py
```

You can open a browser and go to: <http://localhost:5084> and check the software.

## make PreyTouch start on reboot

If the check passed successfully, you may want to make PreyTouch start automatically after reboot. For that you can install supervisorctl, using:

```console
sudo apt install supervisor
```

Create the following configuration file

```console
sudo nano /etc/supervisor/conf.d/prey_touch.conf
```

and put the following inside it:

```ini
[program:prey_touch]
command=<path_to_python_interpreter> api.py
user=<user>
environment=PYTHONUNBUFFERED=1;HOME=/home/<user>
directory=<path_to_PreyTouch_dir>/Arena
stdout_logfile=/var/log/prey_touch.out.log
redirect_stderr=true
killasgroup=true
stopasgroup=true
autorestart=true
```

**Notice!**

- To get the path for the python interpreter you can run "which python" while your python environment is activated.
- Under "directory" you should put before "/Arena" the path to the PreyTocuh dir
- You also need to specify the user which runs PreyTouch

Now you can load this new configuration to supervisor

```console
sudo supervisorctl reread
sudo supervisorctl update
```

and you can always check the logs using:

```console
tail -f /var/log/prey_touch.out.log
```

in case you need to restart PreyTouch:

```console
sudo supervisorctl restart prey_touch
```
