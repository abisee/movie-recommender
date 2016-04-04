from recommend import *

##########################################################################
# PLAY WITH THE CODE BELOW!
##########################################################################

# This part determines which features we will use when comparing movies.
# Change the number to a 1 to use the feature, or 0 to ignore it.
# Currently we only have the "runtime" feature turned on.
features_to_use = {
'year' : 0,
'rating': 0, # imdb rating out of 10
'runtime': 1, # runtime in minutes
'mpaa': 0, # mpaa rating e.g. PG or R
'genres': 0,
'countries': 0,
'languages': 0
}

# Run this function to see the top 5 and bottom 5 matching movies for "Inside Out".
# Try typing in another movie from the database (check titles.txt to see the list of movies).
# You can change the 5 to another number to see more or fewer matches.
# Experiment with turning features on and off to see the effect on the recommender system.
print_top_bottom("Inside Out", features_to_use, 5)

# Run this function to see *all* the movies in the database, along with their
# distance from "Inside Out".
# print_all_dists("Inside Out", features_to_use)

##########################################################################
# END OF YOUR CODE
##########################################################################
