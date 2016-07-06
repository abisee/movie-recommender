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



def html_print(movie_key):
    """Returns a HTML string giving information about the movie.
    Formatted in bullet points"""
    movie = movies[movie_key]

    title_html = heading("%s (%i) (%s)" % (movie['title'], movie['year'], movie['mpaa']),4)

    info = [] # list of strings to go into an unordered list

    info.append("Director: %s" % (movie['director']))
    info.append("Top three cast: %s" % (comma_list(movie['cast'])))
    info.append("Genres: %s" % (comma_list(movie['genres'])))
    info.append("IMDB rating: %.1f/10 from %i votes" % (movie['rating'], movie['votes']))
    info.append("Countries: %s" % (comma_list(movie['countries'])))
    info.append("Languages: %s" % (comma_list(movie['languages'])))
    info.append("Runtime: %i minutes" % (movie['runtime']))
    info.append("Aspect ratio: %s" % (movie['aspect ratio']))
    info.append("Production company: %s" % (movie['production companies']))

    return title_html + unordered_list(info)


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

def html_print_features(feature_weights):
    """HTML-prints information about which features are turned on in feature_weights"""
    header = heading("Using the following features:",3)
    feature_strings = []
    for f in feature_weights.keys():
        weight = feature_weights[f]
        if weight!=0:
            if f=='mpaa':
                f = "MPAA rating"
            elif f=='rating':
                f = "IMDB rating"
            elif f=='production companies':
                f = "production company"
            if weight!=1:
                f += " (weight %.2f)" % (weight)
            feature_strings.append(f)

    return header + unordered_list(feature_strings)




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



def html_top_bottom(movie_key, feature_weights, k, distance=eucl_dist):
    """HTML-prints the top and bottom k movies, excluding the movie itself"""

    # pretty-print the movie and features used
    html_string = ""
    html_string += heading("Finding nearest and furthest neighbors from this movie:",3)
    html_string += html_print(movie_key)
    html_string += html_print_features(feature_weights)
    dists_sorted = get_dists(movie_key, feature_weights, distance)

    html_string += heading("Top %i matching movies" % (k),4)
    dists_sorted.remove((movie_key,0.0)) # remove the movie itself
    for key,dist in dists_sorted[:k]:
        html_string += html_print(key)
    html_string += heading("Bottom %i matching movies" % (k),4)
    for key,dist in dists_sorted[-k:]:
        html_string += html_print(key)
    return HTML(html_string)


def html_all_dists(movie_key, feature_weights, distance=eucl_dist):
    """HTML-prints the distance from all movies to movie_key, from closest to furthest.
    Returns a IPython.core.display.HTML object."""

    # pretty-print the movie and features used
    html_string = ""
    html_string += heading("Finding distances from this movie:",3)
    html_string += html_print(movie_key)
    html_string += html_print_features(feature_weights)
    dists_sorted = get_dists(movie_key,feature_weights,distance)
    dists_sorted = [ ("%.3f"%(dist), key) for (key,dist) in dists_sorted]
    html_string += twocol_table(dists_sorted)

    return HTML(html_string)


def twocol_table(data):
    """Data is a list of pairs.
    Returns a html string"""
    to_return = "<table>"
    for (key,val) in data:
        to_return += "<tr><td>%s</td><td>%s</td></tr>" % (key,val)
    to_return += "</table>"
    return to_return


def unordered_list(lst):
    """takes a python list of strings, returns an html string"""
    to_return = "<ul>"
    for x in lst:
        to_return += "<li>%s</li>" % (x)
    to_return += "</ul>"
    return to_return

def calc_vec_dim():
    """Calculate what the dimension of the vectors should be"""
    d = len(noncategory_features) # five non-categorical features
    for f in category_features:
        d += cat_feat_counts[f]
    return d

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
