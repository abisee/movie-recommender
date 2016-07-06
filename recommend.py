from IPython.core.display import HTML

##########################################################################
# STEP 1: IMPORT THE MOVIES FROM movies.txt
##########################################################################

def import_movies():
    """Returns movies, a dict from title strings to movie.
    Each movie is a dict from feature strings to information (string/integer/list)."""
    movies = {}
    with open('movies.txt') as f:
        for line in f:
            feature_name = line.split(';')[0]
            feature_data = line.split(';')[1][:-1] # cut off the \n

            # This is the first feature so make a new dict
            if feature_name=='title':
                movie = {}

            # Add the feature to the movie
            if len(feature_data.split(','))>1: # list-of-strings features
                movie[feature_name] = feature_data.split(',')[:-1]
            else:
                if feature_name in ['runtime','year','votes']: # integer features
                    movie[feature_name] = int(feature_data)
                elif feature_name=='rating': # float features
                    movie[feature_name] = float(feature_data)
                else: # single string features
                    movie[feature_name] = feature_data

            # This is the last feature so save the movie
            if feature_name=='votes':
                movies[movie['title']]=movie

    return movies

##########################################################################
# STEP 2: COLLECT LISTS OF ALL THE GENRES, COUNTRIES, LANGUAGES, ASPECT RATIOS,
# DIRECTORS, ACTORS, PRODUCTION COMPANIES
##########################################################################


def collect_feat_items(movies,category_features):
    """Collects lists of all category feature values and the number of values.
    Inputs:
        movies: dict from title to movie feature dict
        category_features: list of features to be collected
    Returns:
        cat_feat_items: dict from feature to list of values
        cat_feat_counts: dict from feature to number of values"""
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
    cat_feat_counts = {}
    for f in category_features:
        cat_feat_counts[f] = len(cat_feat_items[f])
    return cat_feat_items, cat_feat_counts


##########################################################################
# STEP 3: CREATE A VECTOR FOR EACH MOVIE.
# Each vector has the format:
# year, rating, runtime, mpaa, votes, [genres], [countries], [languages], [aspect ratio], [director], [cast], [production company]
# One dimension for each: genre, country, language, aspect ratio, director, actor and production company. 0 or 1 in that dimension.
# MPAA rating is converted to a number 0 to 4.
##########################################################################

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

def create_vectors(movies,cat_feat_items):
    """Converts the dictionary representation to a vector representation.
    Inputs:
        movies: dict from title to movie feature dict
        cat_feat_items: dict from category feature to list of possible values
    Returns:
        movie_vectors: dict from titles to vectors (lists)"""

    movie_vectors = {} # will contain vectors
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

    return movie_vectors


##########################################################################
# STEP 4: NORMALIZE THE VECTORS
##########################################################################


def get_means(d,movie_vectors,num_movies):
    """Calculate the mean for each of d dimensions"""
    means = [0.0] * d
    for key,vec in movie_vectors.iteritems():
        for i in range(d):
            means[i] += float(vec[i])
    for i in range(d):
        means[i] /= num_movies
    return means

def get_stds(d,movie_vectors,num_movies,means):
    """Calculate the standard deviation for each of d dimensions"""
    stds = [0.0] * d
    for key,vec in movie_vectors.iteritems():
        for i in range(d):
            stds[i] += (float(vec[i])-means[i])**2
    for i in range(d):
        stds[i] /= num_movies
        stds[i] = stds[i]**0.5
    return stds


def normalize(x,mean,std):
    """Normalizes a scalar x with respect to given mean and std (also scalar)"""
    return (x-mean)/std


def normalize_vec(movie_vec, features_to_normalize, means, stds):
    """Normalizes a single vector according to given means and stds.
    Inputs:
        movie_vec: single movie vec
        features_to_normalize: a dict from feature names to 0 or 1.
            We normalize features marked with 1.
        means and stds: the means and stds over all movie vectors and dimensions
    Returns:
        normalized_vec: movie_vec normalized"""
    d = len(movie_vec)
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


def get_normed_vecs(movie_vectors):
    """Normalizes movie_vectors.
    Returns movie_vectors_norm, a dictionary from title strings to normalized vectors."""
    movie_vectors_norm = {} # will contain normalized vectors

    # Specify which features to normalize
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

    # Calculate the means and stds
    _,any_movie_vec = movie_vectors.popitem() # get any movie...
    d = len(any_movie_vec) # ...to obtain the length
    num_movies = len(movies.keys()) # get the total number of movies
    means = get_means(d,movie_vectors,num_movies)
    stds = get_stds(d,movie_vectors,num_movies,means)

    # Normalize the movie vectors
    for key,vec in movie_vectors.iteritems():
        movie_vectors_norm[key] = normalize_vec(vec,features_to_normalize,means,stds)

    return movie_vectors_norm

##########################################################################
# STEP 5: CODE TO CALCULATE DISTANCES BETWEEN MOVIE VECTORS
##########################################################################

# def eucl_dist(vec1,vec2):
#     """returns euclidean distance between vec1 and vec2, which are both lists of the same length"""
#     dist = 0
#     l = len(vec1)
#     for i in range(l):
#         dist += abs(vec1[i] - vec2[i])**2
#     return dist**0.5
#
# def manh_dist(vec1, vec2):
#     """ vec1 and vec2 are both lists of numbers, with the same length.
#     This function returns the Manhattan distance between vec1 and vec2."""
#     dist = 0
#     l = len(vec1)
#     for i in range(l):
#         dist += abs(vec1[i] - vec2[i])
#     return dist

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
    d = len(movie_vec)
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

def get_dists(movie_key, feature_weights, distance_function):
    """Calculates the distance from all movies in the database to movie_key,
    using the weights given in feature_weights.
    Returns a list of (dist,key) tuples that doesn't include the original movie."""

    # reweight all the vectors
    movie_vecs_reweighted = {}
    for key,vec in movie_vectors_norm.iteritems():
        movie_vecs_reweighted[key] = reweight_vec(vec,feature_weights)
    movie_vec = movie_vecs_reweighted[movie_key]

    # calculate the distances
    dists = []
    for key,vec in movie_vecs_reweighted.iteritems():
        if key!=movie_key:
            dists.append((key, distance_function(movie_vec,vec)))
    dists_sorted = sorted(dists,key=lambda x:x[1]) # sort by second
    return dists_sorted

##########################################################################
# STEP 6: CODE TO PRINT INFORMATION (including HTML)
##########################################################################


def comma_list(lst):
    """takes a python list of strings, returns a string separated by commas"""
    to_return = ""
    for x in lst:
        to_return += "%s, " % (x)
    return to_return[0:-2]

def heading(s,n):
    return "<h%i>%s</h%i>" % (n,s,n)

def table_data(s):
    """puts <td> around input"""
    return "<td>%s</td>" % (s)

def table_header(s):
    """puts <th> around input"""
    return "<th>%s</th>" % (s)

def table_row(data,header=False):
    """Data is a list of strings.
    Returns a HTML string for a table row."""
    to_return = "<tr>"
    for d in data:
        if header:
            to_return += table_header(d)
        else:
            to_return += table_data(d)
    to_return += "</tr>"
    return to_return

def colgroup(feature_weights,feature_order):
    """Returns a HTML string giving the <col> attributes for the table"""
    to_return = "<colgroup>"
    for f in feature_order:
        if f in ['title','distance']:
            to_return += "<col span=\"1\" style=\"background-color:white\">"
        elif feature_weights[f]!=0:
            to_return += "<col span=\"1\" style=\"background-color:white\">" # used
        else:
            to_return += "<col span=\"1\" style=\"background-color:grey\">" # unused
    to_return += "</colgroup>"
    return to_return

def feat_to_info(movie,f,dist):
    """How to format the feature information in the table.
    Returns a string."""
    if f in ['cast','genres','countries','languages']:
        return comma_list(movie[f])
    elif f in ['runtime','year','votes']: # integer features
        return str(movie[f])
    elif f=='rating': # float feature
        return "%.1f" % (movie[f])
    elif f=='distance': # float feature
        return "%.3f" % (dist)
    else:
        return movie[f]


def feat_to_header(f):
    """How to format the feature in the header."""
    if f=='mpaa':
        return "MPAA"
    elif f=='cast':
        return "top 3 cast"
    elif f=='rating':
        return "IMDB rating"
    elif f=='votes':
        return "IMDB votes"
    elif f=='runtime':
        return "runtime (mins)"
    elif f=='production companies':
        return "production company"
    else:
        return f

def table_row_print(movie_key,dist,feature_order):
    """Returns a HTML string giving information about the movie
    as a row of a table. Adds dist (which is a float) to the row."""
    movie = movies[movie_key]
    data = [feat_to_info(movie,f,dist) for f in feature_order]
    return table_row(data)

def get_feature_order(feature_weights):
    """Given the choice of which features to use, determine the column order
    in the table"""
    feature_order = ['distance','title']
    f_w_tuple_list= sorted(feature_weights.iteritems(),key=lambda x:x[1],reverse=True) # sort by weight
    for (f,_) in f_w_tuple_list:
        feature_order.append(f)
    return feature_order

def get_recommendations(movie_key, feature_weights, distance):
    """HTML-prints a table of all movies with features and distance"""
    if movie_key not in movies.keys():
        print 'That film isn\'t in the database. Did you type it correctly?'
        return

    feature_order = get_feature_order(feature_weights)
    dists_sorted = get_dists(movie_key,feature_weights,distance)

    html_string = "<table>"
    html_string += colgroup(feature_weights,feature_order)
    headers = [feat_to_header(f) for f in feature_order]
    html_string += table_row(headers,header=True)
    html_string += table_row_print(movie_key,0,feature_order)
    for (key,dist) in dists_sorted:
        html_string += table_row_print(key,dist,feature_order)
    html_string += "</table>"
    return HTML(html_string)

##########################################################################
# MAIN
##########################################################################

movies = import_movies()

# Noncategory features in the order they appear in the vectors
noncategory_features = ['year', 'rating', 'runtime', 'mpaa', 'votes']
# The category features, in the order they appear in the vectors
category_features = ['genres', 'countries', 'languages', 'aspect ratio', 'director', 'cast', 'production companies']

cat_feat_items,cat_feat_counts = collect_feat_items(movies,category_features)
movie_vectors = create_vectors(movies,cat_feat_items)
movie_vectors_norm = get_normed_vecs(movie_vectors)
