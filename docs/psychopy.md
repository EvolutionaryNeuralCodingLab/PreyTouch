# Integration with PsychoPy

**Notice!** psychopy is not embedded in PreyTouch, and we provide no support on running psychopy experiments. The integration only provides a method to call for external psychopy experiments from within the PreyTouch management.

## Initialization
- We strongly recommend to install psychopy on a different python environment and not to use the same environment of PreyTouch.
- Make sure that psychopy is working properly in that other environment and only then integrate with PreyTouch.
- All psychopy folders should be saved under one main folder
- psychopy folders name should be equal to main python file to run the experiment (e.g. if you have a psychopy folder named ImageStimuli, it must consist of a file called ImageStimuli.py to run the stimulation)
- Make sure that the user running PreyTouch has read and execute permissions to:
  - The python interpreter of the psychopy environment
  - The psychopy folder
- Once, you verified all of the above you may continue for the integration

## Integration
1. Open the settings menu  (click on the cog icon (⚙) on the top-left side of PreyTouch UI)
2. Scroll down and look for PsychoPy
3. fill **PSYCHO_FOLDER** with the full path to the main psychopy folder. 
4. fill **PSYCHO_PYTHON_INTERPRETER** with the full path to the python interpreter. (In linux you can run "which python" from within the environment to get the interpreter path)
5. Restart PreyTouch

In case you need to add some environment variables while executing psychopy please add them before the interpreter name, 
for example in our case we set **PSYCHO_PYTHON_INTERPRETER** to be:
```shell
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 /home/ep-arena/miniconda3/envs/psycho/bin/python
```

## Running a PsychoPy Experiment
If the folder was set correctly you can now see the psychopy experiments in PreyTouch
1. Click on "Create Experiment" button in PreyTouch UI
2. In the block menu change Block Type to "psycho"
3. You should see now a row called "Psycho File" with all psychopy experiments.

![PsychoPy Menu](/docs/images/psychopy_menu.png)
