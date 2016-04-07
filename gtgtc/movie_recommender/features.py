from recommend import *

##########################################################################
# PLAY WITH THE CODE BELOW!
##########################################################################

# This part determines which features we will use when comparing movies.
# Currently we only have the "year" and "runtime" features turned on.
# Change the number to 1 to use the feature, or 0 to ignore it.
# You can even set the weights to values other than 0 or 1!
# For example, set a feature weight to 10 if you want it to be really important,
# or to 0.1 if you want it to be less important (but not zero importance).
feature_weights = {
'year' : 1, # year of release
'rating': 0, # IMDB rating out of 10
'runtime': 1, # runtime in minutes
'mpaa': 0, # MPAA rating e.g. PG or R
'votes': 0, # number of voters on IMDB
'genres': 0,
'countries': 0,
'languages': 0,
'aspect ratio': 0, # ratio of picture size
'director': 0,
'cast': 0, # top three actors
'production companies': 0
}

# Run this function to see the top 5 and bottom 5 matching movies for "Harry Potter and the Goblet of Fire".
# Try typing in another movie from the database (check titles.txt to see the list of movies).
# You can change the 5 to another number to see more or fewer matches.
# Experiment with turning features on and off to see the effect on the recommendations.
print_top_bottom("Harry Potter and the Goblet of Fire", feature_weights, 5)

# Run this function to see *all* the movies in the database, along with their
# distance from "Harry Potter and the Goblet of Fire".
# Again, you can type any movie title in here (that's listed in titles.txt).
# print_all_dists("Harry Potter and the Goblet of Fire", feature_weights)

##########################################################################
# END OF YOUR CODE
##########################################################################
