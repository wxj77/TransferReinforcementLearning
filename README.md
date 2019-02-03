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
  eg. `cp gym path_to_python_lib/python3.6/site-packages/gym`
  eg. `cp atari_py path_to_python_lib/python3.6/site-packages/atari_py`

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

###
