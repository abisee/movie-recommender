##########################################################################
# STEP 1: IMPORT THE MOVIES FROM movies.txt
##########################################################################

movies = {} # will contain movies

with open('movies.txt') as f:
    for line in f:
        feature_name = line.split(';')[0]
        feature_data = line.split(';')[1][:-1] # without the \n
        if feature_name=='title':
            # make a new dictionary
            movie = {}
        # add this feature to movie
        if len(feature_data.split(','))>1: # list-of-strings features
            movie[feature_name] = feature_data.split(',')[:-1]
        else:
            if feature_name=='runtime' or feature_name=='year' or feature_name=='votes': # integer features
                movie[feature_name] = int(feature_data)
            elif feature_name=='rating': # float features
                movie[feature_name] = float(feature_data)
            else: # string features
                movie[feature_name] = feature_data
        if feature_name=='votes':
            # save it
            movies[movie['title']]=movie

num_movies = len(movies.keys())

# for title in sorted(movies.keys()):
#     print title

##########################################################################
# STEP 2: COLLECT LISTS OF ALL THE GENRES, COUNTRIES, LANGUAGES, ASPECT RATIOS,
# DIRECTORS, ACTORS, PRODUCTION COMPANIES
##########################################################################

noncategory_features = ['year', 'rating', 'runtime', 'mpaa', 'votes']
category_features = ['genres', 'countries', 'languages', 'aspect ratio', 'director', 'cast', 'production companies']
cat_feat_items = {}

for f in category_features:
    cat_feat_items[f] = []

for _,movie in movies.iteritems():
    for f in category_features:
        if f=='genres' or f=='countries' or f=='languages' or f=='cast': # if a list
            for item in movie[f]:
                if item not in cat_feat_items[f]:
                    cat_feat_items[f].append(item)
        else: # single
            item = movie[f]
            if item not in cat_feat_items[f]:
                cat_feat_items[f].append(item)

# for f in category_features:
#     print f, cat_feat_items[f]
#     print ""

cat_feat_counts = {}

for f in category_features:
    cat_feat_counts[f] = len(cat_feat_items[f])

# print cat_feat_counts

##########################################################################
# STEP 3: CREATE A VECTOR FOR EACH MOVIE.
# Each vector has the format:
# year, rating, runtime, mpaa, votes, [genres], [countries], [languages], [aspect ratio], [director], [cast], [production company]
# One dimension for each: genre, country, language, aspect ratio, director, actor and production company. 0 or 1 in that dimension.
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
    vec.append(movie['votes'])
    for f in category_features:
        for item in cat_feat_items[f]:
            if f=='genres' or f=='countries' or f=='languages' or f=='cast': # if a list
                if item in movie[f]:
                    vec.append(1)
                else:
                    vec.append(0)
            else: # single, like aspect ratio, director or production company
                if movie[f]==item:
                    vec.append(1)
                else:
                    vec.append(0)
    # save it
    movie_vectors[key] = vec


##########################################################################
# STEP 4: NORMALIZE THE VECTORS
##########################################################################

# calculate the dimension d of the vectors
d = len(noncategory_features) # five non-categorical features
for f in category_features:
    d += cat_feat_counts[f]

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
    """normalizes a datapoint x with respect to given mean and std"""
    return (x-mean)/std

def normalize_vec(movie_vec, features_to_normalize):
    normalized_vec = [0.0] * d
    for idx,f in enumerate(noncategory_features): # non category features
        if features_to_normalize[f] == 1:
            normalized_vec[idx] = normalize(movie_vec[idx],means[idx],stds[idx])
        else:
            normalized_vec[idx] = movie_vec[idx]
    start_idx = len(noncategory_features) # category features
    for f in category_features:
        length = cat_feat_counts[f]
        for idx in range(length):
            if features_to_normalize[f] == 1:
                normalized_vec[start_idx + idx] = normalize(movie_vec[start_idx+idx],means[start_idx+idx],stds[start_idx+idx])
            else:
                normalized_vec[start_idx + idx] = movie_vec[start_idx+idx]
        start_idx = start_idx + length
    return normalized_vec

# specify which features to normalize
features_to_normalize = {
'year' : 1, # year of release
'rating': 1, # IMDB rating out of 10
'runtime': 1, # runtime in minutes
'mpaa': 1, # MPAA rating e.g. PG or R
'votes': 1, # number of voters on IMDB
'genres': 1,
'countries': 1,
'languages': 1,
'aspect ratio': 1, # ratio of picture size
'director': 0,
'cast': 0, # top three actors
'production companies': 1
}

# normalize the movie vectors
for key,vec in movie_vectors.iteritems():
    movie_vectors_norm[key] = normalize_vec(vec,features_to_normalize)


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

def list_subtract(vec1,vec2):
    """vec1 and vec2 are lists of the same length, containing numbers.
    Returns vec1-vec2 as a list"""
    answer = []
    l = len(vec1)
    for i in range(l):
        answer.append(vec1[i]-vec2[i])
    return answer

def reweight_vec(movie_vec, feature_weights):
    """Returns a reweighted version of movie_vec according to the weights in feature_weights.
    The returned vector is same dimension as movie_vec"""
    reweighted_vec = [0.0] * d
    for idx,f in enumerate(noncategory_features): # non category features
        reweighted_vec[idx] = feature_weights[f] * movie_vec[idx]
    start_idx = len(noncategory_features) # category features
    for f in category_features:
        length = cat_feat_counts[f]
        for idx in range(length):
            reweighted_vec[start_idx + idx] = feature_weights[f] * movie_vec[start_idx + idx]
        start_idx = start_idx + length
    return reweighted_vec


def pretty_print(movie_key):
    """pretty-prints information about the movie"""
    movie = movies[movie_key]
    print "  %s (%i) (%s)" % (movie['title'], movie['year'], movie['mpaa'])
    print "  Director:           ", movie['director']
    print "  Top three cast:     ",
    for a in movie['cast']:
        print a+',',
    print ""
    print "  Genres:             ",
    for g in movie['genres']:
        print g,
    print ""
    print "  IMDB rating:         %.1f/10 from %i votes" % (movie['rating'], movie['votes'])
    print "  Countries:          ",
    for c in movie['countries']:
        print c,
    print ""
    print "  Languages:          ",
    for l in movie['languages']:
        print l,
    print ""
    print "  Runtime:             %i minutes" % (movie['runtime'])
    print "  Aspect ratio:       ", movie['aspect ratio']
    print "  Production company: ", movie['production companies']


def pretty_print_features(feature_weights):
    """Pretty-prints information about which features are turned on in feature_weights"""
    print "Comparing movies with respect to the following features: "
    for f in feature_weights.keys():
        weight = feature_weights[f]
        if weight!=0:
            if f=='mpaa':
                print " > MPAA rating",
            elif f=='rating':
                print " > IMDB rating",
            elif f=='production companies':
                print " > production company",
            else:
                print " > %s" % (f),
            if weight!=1:
                print "(weight %.2f)" % (weight)
            else:
                print ""

def pretty_print_vec(movie_vec):
    """for debugging. prints the vector along with corresponding features"""
    for idx,f in enumerate(noncategory_features): # non category features
        if movie_vec[idx]!=0:
            print "%.3f %s" % (movie_vec[idx], f)
    start_idx = len(noncategory_features) # category features
    for f in category_features:
        for idx,item in enumerate(cat_feat_items[f]):
            if movie_vec[start_idx + idx]!=0:
                print "%.3f %s" % (movie_vec[start_idx + idx], item)
        start_idx = start_idx + cat_feat_counts[f]


def get_dists(movie_key, feature_weights, distance_function):
    """Calculates the distance from all movies in the database to movie_key,
    using the weights given in feature_weights"""
    # reweight all the vectors
    movie_vecs_reweighted = {}
    for key,vec in movie_vectors_norm.iteritems():
        movie_vecs_reweighted[key] = reweight_vec(vec,feature_weights)

    movie_vec = movie_vecs_reweighted[movie_key]
    dists = []
    for key,vec in movie_vecs_reweighted.iteritems():
        dists.append((key, distance_function(movie_vec,vec)))
    dists_sorted = sorted(dists,key=lambda x:x[1]) # sort by second
    return dists_sorted


def print_all_dists(movie_key, feature_weights, distance=eucl_dist):
    """Prints the distance from all movies to movie_key, from closest to furthest"""

    # pretty-print the movie and features used
    print "=============================================================="
    print "Finding distances from movies to this movie:"
    print "=============================================================="
    pretty_print(movie_key)
    print "=============================================================="
    pretty_print_features(feature_weights)
    print "=============================================================="

    dists_sorted = get_dists(movie_key,feature_weights,distance)
    for key,dist in dists_sorted:
        print "%.3f %s" % (dist,key)


def print_top_bottom(movie_key, feature_weights, k, distance=eucl_dist):
    """Pretty-prints the top and bottom k movies, excluding the movie itself"""

    # pretty-print the movie and features used
    print "=============================================================="
    print "Finding nearest and furthest neighbors from this movie:"
    pretty_print(movie_key)
    print "=============================================================="
    pretty_print_features(feature_weights)
    print "=============================================================="

    dists_sorted = get_dists(movie_key, feature_weights, distance)
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
# ADVANCED CODING EXERCISE
# If you already know some Python, have a go at this!
##########################################################################

# Implement the following function to compute the Manhattan distance
# (discussed in the lecture) between two vectors.
# For a hint on how to write this function, look at the function eucl_dist
# further up this document.
# Once you've completed this function, go to features.py and add "eucl_dist"
# as the final argument to the functions print_top_bottom and print_all_dists.
# Then the recommender system will be using the Manhattan distance instead!
def manh_dist(vec1, vec2):
    """ vec1 and vec2 are both lists of numbers, with the same length.
    This function returns the Manhattan distance between vec1 and vec2."""
    return 0.0
