# Unity - Python Reinforcement Learning

![Example](/documents/environment.gif)

## Install dependencies
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
If your agent made it through the whole track, a ppo_actor.h5 and a ppo_critic.h5 file will be created. The test script will generate a video visualizing how your agent would reacted given real "markku" images (https://markku.ai/).

1. check that a valid model.h5 file is available
1. navigate to ```cd ./python-reinforcement-learning```
1. start the virtualenv ```pipenv shell```
1. run script ```python3.7 ./src/test.py```

## Run the game headless
1. build the game using unity (name of the app in my case is "game-build")
1. open comand line and navigate to ```cd <path-to-build>/game-build.app/Contents/MacOS```
1. run the game in batch mode ```./game-build -batchmode```

## Create a frame by frame recording
1. navigate to ```cd ./python-reinforcement-learning```
1. start the virtualenv ```pipenv shell```
1. run the recording script ```python ./src/recording.py```
1. open the unity project and start the game
1. the recoding will be saved as soon as the goal is reached or you are off track

## Next Up
1. give higher rewards to agents which complete the track with the least steps
1. add additional exploraton strategies such as RND or self-supervised prediction
- https://towardsdatascience.com/explained-curiosity-driven-learning-in-rl-exploration-by-random-network-distillation-72b18e69eb1b
- https://medium.com/data-from-the-trenches/curiosity-driven-learning-through-next-state-prediction-f7f4e2f592fa

## Analyse the model using tensorboard (Not Implemented Yet)
1. navigate to ```cd ./python-reinforcement-learning```
1. start the virtualenv ```pipenv shell```
1. run ```tensorboard --logdir ./logs```
1. start agent training
1. open web interface ```http://localhost:6006/```