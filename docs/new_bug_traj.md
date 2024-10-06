# Adding New Prey Movement Profile

Currently, the application supports the following bug movements:
- horizontal movement (hole to hole)
- circular movement
- half-circular movement
- random movement (random straight lines)
- static movement (bug/object doesn't move)

If the user wants to add a new bug movement:
1. Open the application config file in ```pogona_hunter/src/config.json```, and add the name of the new movement under: boards > holes
2. Open the bug vue file in ```pogona_hunter/src/components/bugs/holesBug.vue```
3. The logic of this movement must be integrated into 2 main functions:
   1. move() - this function is called for each frame animation of the application
   2. initiateStartPosition() - this is the initialization of the bug.
4. Notice that the holeBug module gets the movement type from ```this.bugsSettings.movementType``` use that with the new name you added to config.json
5. Rebuild the application docker service ```cd docker && docker-compose up -d --build app```
