# KapiBara ROS2 container

Container with ROS2 code for KapiBara robot. It handles development and retail container for deployment on real hardware.

## Development containter

To run dev container execute:

```bash
./dev.sh start
```

Container should start then in seperate terminal you can execute any ros command using:

```bash
./dev.sh cmd \<command\>
```

## Production containter

First build containter execute:

``` bash
./prod.sh build
```

Then execute:

``` bash
./prod.sh start_bg
```

## Helper commands

There is a bunch of helper scripts prepared you can acess them by typing:

``` bash
./(prod.sh | dev.sh) cmd /cmd/\<command name\>
```

Commands are in commands folder with descriptions.

## Dependencies

In folder dependencies there is a file called packages.txt which holds
linux packages required for container to run.

## Environment variables

env folder holds two files **dev** and **prod**. **dev** file holds envrionment variables
for development container and **prod** holds variables for production container.

## Workspace

It is a workspace folder in which every ROS 2 packages go.

## Other containers

- Container with HAL - <https://github.com/pperonski/KapiBara-ROS-2-HAL>
- Container with Mojoco simulation - <https://github.com/pperonski/KapiBara-ROS-2-Mujoco>

## Run

### Run in simulation

In order to run simulation with KapiBara robot, clone repository:

``` bash
git clone https://github.com/pperonski/KapiBara-ROS-2-Mujoco mujoco
```

Go to simulation directory:

``` bash
cd mujoco
```

Build container:

``` bash
./build.sh
```

Then run container:

``` bash
./run.sh
```

Then in main repository execute:

``` bash
./dev.sh start
```

### Run on hardware

In order to run container on real robot clone:

``` bash
git clone https://github.com/pperonski/KapiBara-ROS-2-HAL hal
```

Go to hardware directory:

``` bash
cd hal
```

Build container:

``` bash
./run.sh build
```

Run container:

``` bash
./run.sh start
```

Then in main repository execute:

``` bash
./dev.sh start
```
