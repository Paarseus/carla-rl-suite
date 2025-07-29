

number_of_vehicles = 50
number_of_walkers = 5

safe = True     #Avoid spawning vehicles prone to accidents

filterv = "vehicle.*"    #Filter vehicle model (default: "vehicle.*")
generationv = "all"        #restrict to certain vehicle generation (values: "2","3","All" - default: "All")
filterw = "walker.pedestrian.*"    #Filter pedestrian type (default: "walker.pedestrian.*")
generationw = "all"         #restrict to certain pedestrian generation (values: "2","3","All" - default: "All")
seed = None
seedw = None

tm_port = 8000     #Port to communicate with TM (default: 8000)'

asynch = False      #Activate asynchronous mode execution
hybrid = False      #Activate hybrid mode for Traffic Manager

car_lights_on = False   #Enable automatic car light management
hero = False    #Set one of the vehicles as hero
respawn = False     #Automatically respawn dormant vehicles (only in large maps)
no_rendering = False    #Activate no rendering mode
