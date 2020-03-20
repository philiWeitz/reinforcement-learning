# Unity - Python Reinforcement Learning

![Example](/documents/example.gif)

## install dependencies
1. navigate to ```cd ./python-reinforcement-learning```
1. start a new virtualenv ```pipenv shell```
1. install all dependencies ```pipenv install```

## Start python
1. navigate to ```cd ./python-reinforcement-learning```
1. start the virtualenv ```pipenv shell```
1. run script ```python3.7 ./src/unity-connector.py```

## Start unity
1. open unity project
1. load RaceTrack scene
1. run scene

## Testing your model against real data
If your agent made it through the whoole track, a model.h5 file will be created. The test script will generate a video visualizing how your agent would have reacted given real "markkuu" images.

1. check that a valid model.h5 file is available
1. navigate to ```cd ./python-reinforcement-learning```
1. start the virtualenv ```pipenv shell```
1. run script ```python3.7 ./test/test.py```