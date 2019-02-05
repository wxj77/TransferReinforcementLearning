# TransferReinforcementLearning

---

### Setup

The current code works on python 3.6
* install the require python packages
  * gym gym\[atari\] tensorflow
    see instructions from [openai](https://gym.openai.com/), [tensorflow](https://www.tensorflow.org/)

* clone the repositary
  `git clone https://github.com/wxj77/TransferReinforcementLearning.git`
  `cd TransferReinforcementLearning`

* (optional) copy "gym" and "atari_py" folders to your python package directory, which allows you to use rotated atari pong game.
  eg. `cp -r gym path_to_python_lib/python3.6/site-packages/gym`
  (obsolete) eg. `cp -r atari_py path_to_python_lib/python3.6/site-packages/atari_py`

* (optional) get tensorpack which 

* Get a solution for "pong game".
  `python pongSimpleSol.py`

* Get a solution for "breakout game" with transfer solutions from "pong game".
  `python breakoutTransferSol.py` 

---

### Files

src: code
  pongSimpleFunc.py
  compareMultiThread.py   

results: results from this project
gym: my custom revision of gym from [openai](https://github.com/openai/gym.git).
atari\_py: my custom revision of atari\_py from [openai atari_py](https://github.com/openai/atari-py.git)

---

### Reference
https://github.com/tensorpack/tensorpack.git
https://github.com/tensorpack/tensorpack/tree/master/examples/DeepQNetwork/

### Tricks
./tensorpack/examples/DeepQNetwork/DQN.py --env ~/anaconda3/lib/python3.6/site-packages/atari_py/atari_roms/breakout.bin --task play --load DoubleDQN-Breakout.npz 


./tensorpack/examples/DeepQNetwork/DQN.py --env ./breakout.bin --task play --load DoubleDQN-Breakout.npz 
./tensorpack/examples/DeepQNetwork/DQN_new.py --env ./pong.bin --task play --load DoubleDQN-Breakout.npz 


pip install opencv-python
pip install ale_python_interface


pretrained models
http://models.tensorpack.com/OpenAIGym/



# Tricks ffmpeg
ffmpeg -i input.mkv -codec copy output.mp4

mkdir frames
#ffmpeg -i input.mp4 -vf scale=320:-1:flags=lanczos,fps=10 frames/ffout%05d.png
ffmpeg -i input.mp4 -vf fps=10 frames/ffout%05d.png
convert -delay 5 -loop 0 frames/ffout*.png output.gif

# References
A3C: https://arxiv.org/pdf/1602.01783.pdf


./tensorpack/examples/A3C-Gym/train-atari.py --env Pong-v0 --task play --load Breakout-v0.npz --task dump_video --output . --episode 1



