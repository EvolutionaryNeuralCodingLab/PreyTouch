## Adding New Bug
1. Generate a set of a minimum of three images capturing a bug, wherein the bug exhibits slight movement across consecutive frames. 

![Ant Images](/docs/images/new_bug_images.png)

2. Images must be 200x200 pixels and saved with the following pattern: <bug_name><id>.png (index starts with 0). For example the file names for the above example should be: ant0.png, ant1.png, … , ant5.png
3. Pick one of the images and apply a blood/plasma stain onto it. This image should be named: <bug_name>_dead.png (e.g. ant_dead.png)

![Dead ant](/docs/images/new_bug_dead.png)

4. Save all the bug images under pogona_hunter/src/assets
5. Open the file pogona_hunter/src/config.json and add the following dictionary:
```json
"ant": {
      "text": "Ant",
      "numImagesPerBug": 6,
      "speed": 3,
      "radiusRange": {
        "min": 120,
        "max": 130
      },
      "stepsPerImage": 5
    }
```
“stepsPerImage” - the number of screen frames to set for each bug image. For example if the screen’s refresh rate is 60Hz and stepsPerImage=5 , then each bug image appears for 5*(1/60) ~ 83 milliseconds.
“speed” and “radiusRange” specify the default values for this bug in case they aren’t given in an experiment.

6. Rebuild the app container

