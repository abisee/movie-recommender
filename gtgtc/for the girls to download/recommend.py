##########################################################################
# STEP 1: IMPORT THE MOVIES FROM movies.txt
##########################################################################

movies = {} # will contain movies

with open('movies.txt') as f:
    for line in f:
        feature_name = line.split(';')[0]
        feature_data = line.split(';')[1][:-1] # without the \n
        if feature_name=='title':
            # make a new dict
            movie = {}
        # add this feature to movie
        if len(feature_data.split(','))>1:
            movie[feature_name] = feature_data.split(',')[:-1]
        else:
            if feature_name=='runtime' or feature_name=='year':
                movie[feature_name] = int(feature_data)
            elif feature_name=='rating':
                movie[feature_name] = float(feature_data)
            else:
                movie[feature_name] = feature_data
        if feature_name=='rating':
            # save this movie to d
            movies[movie['title']]=movie

num_movies = len(movies.keys())

##########################################################################
# STEP 2: COLLECT LISTS OF ALL THE GENRES, COUNTRIES AND LANGUAGES
##########################################################################

genres = []
countries = []
languages = []

for _,movie in movies.iteritems():
    for g in movie['genres']:
        if g not in genres:
            genres.append(g)
    for c in movie['countries']:
        if c not in countries:
            countries.append(c)
    for l in movie['languages']:
        if l not in languages:
            languages.append(l)

num_genres = len(genres)
num_countries = len(countries)
num_languages = len(languages)

##########################################################################
# STEP 3: CREATE A VECTOR FOR EACH MOVIE.
# Each vector has the format: year, rating, runtime, mpaa, [genres], [countries], [languages]
# One dimension for each genre, each country and each language. 0 or 1 in that dimension.
# MPAA rating is converted to a number 0 to 4.
##########################################################################


movie_vectors = {} # will contain vectors

def mpaa_to_num(mpaa_rating):
    """converts MPAA rating to number 0 to 4"""
    if mpaa_rating=='G':
        return 0
    elif mpaa_rating=='PG':
        return 1
    elif mpaa_rating=='PG-13':
        return 2
    elif mpaa_rating=='R':
        return 3
    elif mpaa_rating=='NC-17':
        return 4
    else:
        raise ValueError('unknown rating!' + mpaa_rating)

for key,movie in movies.iteritems():
    vec = []
    vec.append(movie['year'])
    vec.append(movie['rating'])
    vec.append(movie['runtime'])
    vec.append(mpaa_to_num(movie['mpaa']))
    for g in genres:
        if g in movie['genres']:
            vec.append(1)
        else:
            vec.append(0)
    for c in countries:
        if c in movie['countries']:
            vec.append(1)
        else:
            vec.append(0)
    for l in languages:
        if l in movie['languages']:
            vec.append(1)
        else:
            vec.append(0)
    movie_vectors[key] = vec

##########################################################################
# STEP 4: NORMALIZE THE VECTORS
##########################################################################

d = 4 + num_genres + num_countries + num_languages # dimension of the vectors

# calculate the mean of each dimension
means = [0.0] * d
for key,vec in movie_vectors.iteritems():
    for i in range(d):
        means[i] += float(vec[i])
for i in range(d):
    means[i] /= num_movies

# calculate the std of each dimension
stds = [0.0] * d
for key,vec in movie_vectors.iteritems():
    for i in range(d):
        stds[i] += (float(vec[i])-means[i])**2
for i in range(d):
    stds[i] /= num_movies
    stds[i] = stds[i]**0.5

movie_vectors_norm = {} # will contain normalized vectors

def normalize(x,mean,std):
    return (x-mean)/std

# normalize the movie vectors
for key,vec in movie_vectors.iteritems():
    norm_vec = [0.0]*d
    for i in range(d):
        norm_vec[i] = normalize(vec[i],means[i],stds[i])
    movie_vectors_norm[key] = norm_vec

##########################################################################
# STEP 5: CODE TO CALCULATE DISTANCES BETWEEN MOVIE VECTORS
##########################################################################

def eucl_dist(vec1,vec2):
    """returns euclidean distance between vec1 and vec2, which are both lists of the same length"""
    dist = 0
    l = len(vec1)
    for i in range(l):
        dist += abs(vec1[i] - vec2[i])**2
    return dist**0.5

def filter_vec(movie_vec, features_to_use):
    """Returns a filtered vec containing only the features in features_to_use.
    movie_vec is an unfiltered normalized movie vector containing all features."""
    filtered_vec = []
    if features_to_use['year']==1:
        filtered_vec.append(movie_vec[0])
    if features_to_use['rating']==1:
        filtered_vec.append(movie_vec[1])
    if features_to_use['runtime']==1:
        filtered_vec.append(movie_vec[2])
    if features_to_use['mpaa']==1:
        filtered_vec.append(movie_vec[3])
    if features_to_use['genres']==1:
        filtered_vec = filtered_vec + movie_vec[4:4+num_genres]
    if features_to_use['countries']==1:
        filtered_vec = filtered_vec + movie_vec[4+num_genres:4+num_genres+num_countries]
    if features_to_use['languages']==1:
        filtered_vec = filtered_vec + movie_vec[4+num_genres+num_countries:]
    return filtered_vec


def pretty_print(movie_key):
    """pretty-prints information about the movie"""
    movie = movies[movie_key]
    print "  %s (%i) (%s)" % (movie['title'], movie['year'], movie['mpaa'])
    print "  Genres: ",
    for g in movie['genres']:
        print g,
    print ""
    print "  IMDB rating: %.1f/10" % (movie['rating'])
    print "  Countries: ",
    for c in movie['countries']:
        print c,
    print ""
    print "  Languages: ",
    for l in movie['languages']:
        print l,
    print ""
    print "  Runtime: %i" % (movie['runtime'])

def pretty_print_features(features_to_use):
    """Pretty-prints information about which features are turned on in features_to_use"""
    print "Comparing movies with respect to the following features: "
    for f in features_to_use.keys():
        if features_to_use[f]==1:
            if f=='mpaa':
                print " > MPAA rating"
            elif f=='rating':
                print " > IMDB rating"
            else:
                print " > %s" % (f)


def get_dists(movie_key,features_to_use):
    """Calculates the distance from all features in the database to movie_key,
    using only the features turned on in features_to_use"""
    # filter all the vecs
    movie_vecs_filtered = {}
    for key,vec in movie_vectors_norm.iteritems():
        movie_vecs_filtered[key] = filter_vec(vec,features_to_use)

    movie_vec = movie_vecs_filtered[movie_key]
    # print movie_vec
    dists = []
    for key,vec in movie_vecs_filtered.iteritems():
        dists.append((key,eucl_dist(movie_vec,vec)))
    dists_sorted = sorted(dists,key=lambda x:x[1]) # sort by second
    return dists_sorted


def print_all_dists(movie_key, features_to_use):
    """Prints the distance from all movies to movie_key, from closest to furthest"""

    # pretty-print the movie and features used
    print "=============================================================="
    print "Finding distances from movies to this movie:"
    print "=============================================================="
    pretty_print(movie_key)
    print "=============================================================="
    pretty_print_features(features_to_use)
    print "=============================================================="

    dists_sorted = get_dists(movie_key,features_to_use)
    for key,dist in dists_sorted:
        print "%.3f %s" % (dist,key)


def print_top_bottom(movie_key,features_to_use,k):
    """Pretty-prints the top and bottom k movies, excluding the movie itself"""

    # pretty-print the movie and features used
    print "=============================================================="
    print "Finding nearest and furthest neighbors from this movie:"
    pretty_print(movie_key)
    print "=============================================================="
    pretty_print_features(features_to_use)
    print "=============================================================="

    dists_sorted = get_dists(movie_key,features_to_use)
    print "Top %i matching movies" % (k)
    print "=============================================================="
    dists_sorted.remove((movie_key,0.0)) # remove the movie itself
    for key,dist in dists_sorted[:k]:
        pretty_print(key)
        print ""
    print "=============================================================="
    print "Bottom %i matching movies" % (k)
    print "=============================================================="
    for key,dist in dists_sorted[-k:]:
        pretty_print(key)
        print ""
    print "=============================================================="


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
