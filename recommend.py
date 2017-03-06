from IPython.core.display import HTML
import json
import numpy

data_file = "data/data.json"

# Numerical features in the order they appear in the vectors
numerical_features = ['year', 'rating', 'runtime', 'mpaa', 'votes']

# Category features (i.e. one dimension per genre/country/actor), in the order they appear in the vectors
category_features = ['genres', 'countries', 'languages', 'aspect ratio', 'director', 'cast', 'production companies', 'cinematographer', 'kind', 'original music', 'producer', 'writer', 'keywords']

dont_normalize = ['director', 'cast', 'cinematographer', 'original music', 'producer', 'writer', 'keywords']

# Dictionary features (i.e. the feature maps to a dictionary, each of whose keys is one dimension in the vector) in the order they appear in the vectors
dictionary_features = ['demographic', 'parents guide']

# in the dataset, "demographic" maps to a dictionary which maps the demographic names to the [votes, rating] list length 2

demographic_keys = ['aged 18-29', 'aged 30-44', 'aged 45+', 'aged under 18', 'all votes', 'females', 'females aged 18-29', 'females aged 30-44', 'females aged 45+', 'females under 18', 'imdb staff', 'males', 'males aged 18-29', 'males aged 30-44', 'males aged 45+', 'males under 18', 'non-us users', 'top 1000 voters', 'us users']

age_brackets = ['aged under 18', 'aged 18-29', 'aged 30-44', 'aged 45+']

demographic_features = ['fraction votes female', 'fraction votes non-US', 'age bracket with most votes']

# In the data, "parents guide" maps to a dictionary mapping e.g. "violence & gore" to a list of sentences. Sometimes the first sentence is e.g. "2/10" plus occasionally some more text. Can assume that if "2/10" appears it is the start of the first sentence.
parents_guide_features = ['alcohol/drugs/smoking', 'frightening/intense scenes', 'profanity', 'sex & nudity', 'violence & gore']

##########################################################################
# STEP 1: IMPORT THE MOVIES FROM data/data.json
##########################################################################

def import_movies():
  """Returns movies, a dict from title strings to movie. Each movie is a dict from feature strings to information (string/integer/list)."""
  title2movie = {}
  with open(data_file) as f:
    data = json.load(f) # a list
  for movie in data:
    title2movie[movie['title']] = movie
  return title2movie


##########################################################################
# STEP 1.5: PROCESS SOME VALUES
# Convert MPAA strings to numerical values
# Convert dictionary features to individual features
##########################################################################

def mpaa_to_num(mpaa_rating):
  """Converts MPAA rating string to number 0 to 4"""
  if 'Rated G' in mpaa_rating:
    return 0
  elif 'Rated PG' in mpaa_rating:
    return 1
  elif 'Rated PG-13' in mpaa_rating:
    return 2
  elif 'Rated R' in mpaa_rating:
    return 3
  elif 'Rated NC-17' in mpaa_rating:
    return 4
  else:
    raise ValueError("Unknown rating: %s" % mpaa_rating)


def age_bracket_to_num(age_bracket):
  """Returns age bracket is a number 0 to 3"""
  if age_bracket not in age_brackets:
    raise ValueError("Unknown age bracket: %s" % age_bracket)
  return age_brackets.index(age_bracket)


def postprocess_movies(title2movie):
  for _, movie in title2movie.iteritems(): # movie is a dict
    if 'mpaa' in movie.keys():
      movie['mpaa'] = mpaa_to_num(movie['mpaa'])
    if 'demographic' in movie.keys():
      dems = movie['demographic']
      assert demographic_keys == sorted(dems.keys())
      votes = {k: float(v[0]) for k,v in dems.iteritems()}
      ratings = {k: v[1] for k,v in dems.iteritems()}
      gender_feat = votes['females'] / (votes['females'] + votes['males']) # fraction of votes female
      nonus_feat = votes['non-us users'] / (votes['non-us users'] + votes['us users']) # fraction of votes non-us
      ages_votes = sorted([(a,votes[a]) for a in age_brackets], key=lambda (a,v): v)
      age_mode = ages_votes[-1][0] # string of the age bracket with highest votes
      age_feat = age_bracket_to_num(age_mode) # convert to integer 0 to 3

      # print movie['title']
      # print "gender_feat: ", gender_feat
      # print "nonus_feat: ", nonus_feat
      # print "age_feat: ", age_feat
      # print ""

      movie[demographic_features[0]] = gender_feat
      movie[demographic_features[1]] = nonus_feat
      movie[demographic_features[2]] = age_feat
    if 'parents guide' in movie.keys():
      pg = movie['parents guide']
      for f in parents_guide_features:
        if f not in pg.keys(): continue
        fst_sent = pg[f][0]
        if "/10" in fst_sent:
          rating = fst_sent.split("/10")[0].split(" ")[-1]
          try:
            rating = float(rating)
          except:
            print fst_sent
            print rating
            print ""
            continue
          movie[f] = rating
  return title2movie


##########################################################################
# STEP 2: COLLECT LISTS OF ALL THE GENRES, COUNTRIES, LANGUAGES, ASPECT RATIOS,
# DIRECTORS, ACTORS, PRODUCTION COMPANIES
##########################################################################


def collect_feat_items(title2movie, category_features):
  """Collects lists of all category feature values and the number of possible values.
  Inputs:
    title2movie: dict from title to movie feature dict
    category_features: list of features to be collected
  Returns:
    cat_feat_items: dict from feature to list of possible values
    cat_feat_counts: dict from feature to number of possible values"""
  cat_feat_items = {f: [] for f in category_features}

  for _,movie in title2movie.iteritems():
    for f in category_features:
      if f not in movie.keys(): continue
      if type(movie[f])==list:
        for item in movie[f]:
          if item not in cat_feat_items[f]:
            cat_feat_items[f].append(item)
      else: # single
        item = movie[f]
        if item not in cat_feat_items[f]:
          cat_feat_items[f].append(item)

  for f, values in cat_feat_items.iteritems():
    print f, len(values)
    raw_input("press enter to see next...")
    for v in sorted(values):
      print v
    print ""
    raw_input("press enter to see next...")

  cat_feat_counts = {f: len(cat_feat_items[f]) for f in category_features}
  return cat_feat_items, cat_feat_counts


##########################################################################
# STEP 3: CREATE A VECTOR FOR EACH MOVIE.
# Each vector has the format: <all the numerical features>, [demographic_features], [parents_guide_features], <all the categorical features>
# Categorical features have one dimension per e.g. actor, with a 0 or 1 value
##########################################################################

class MovieVec(Object):
  """Vector object to hold movie information"""

  def __init__(movie, cat_feat_items):
    """Movie is a dictionary."""
    vec = []
    feats2indices = {} # maps feature name to list of indices (in the vector)

    for f in numerical_features + demographic_features + parents_guide_features:
      vec.append(movie[f])
      feats2indices[f] = [vec.size]

    for f in category_features:
      num_items = len(cat_feat_items[f])
      feats2indices[f] = range(vec.size+1, vec.size+1+num_items)
      for item in cat_feat_items[f]:
        if type(movie[f])==list:
          if item in movie[f]:
            vec.append(1)
          else:
            vec.append(0)
        else: # single, like aspect ratio, director or production company
          if movie[f]==item:
            vec.append(1)
          else:
            vec.append(0)
    self.vec = np.array(vec)
    self.feats2indices = feats2indices

  def dim(self):
    return self.vec.size

  def normalize(self, means, stds):
    """Subtracts the mean then divides by the std elementwise.
    TODO: change this so that it doesn't normalize some dimensions"""
    assert means.shape == self.vec.shape
    assert stds.shape == self.vec.shape
    assert all(np.nonzero(stds))
    self.vec = (self.vec - means) / stds

  def reweight(self, feature_weights):
    """Reweights according to feature_weights, which is a dictionary from features to weights.
    In particular feature_weights has keys numerical_features + demographic_features + parents_guide_features + categorical_features, which should be the same keyset as self.feats2indices."""
    d = self.dim()
    for f, wt in feature_weights.iteritems():
      idx_span = self.feats2indices[f]
      for idx in idx_span:
        self.vec[idx] = self.vec[idx] * wt


def create_vectors(title2movie, cat_feat_items):
  """Converts the dictionary representation to a vector representation.
  Inputs:
    title2movie: dict from title to movie feature dict
    cat_feat_items: dict from category feature to list of possible values
  Returns:
    title2MVec: dict from titles to vectors (lists)"""

  title2MVec = {} # will contain vectors
  for key,movie in title2movie.iteritems():
    title2MVec[key] = MovieVec(movie, cat_feat_items)
  return title2MVec


##########################################################################
# STEP 4: NORMALIZE THE VECTORS
##########################################################################


def get_means(d, title2MVec, num_movies):
  """title2MVec is a dictionary from titles to MovieVecs.
  d is the dimension of all the MovieVecs.
  Return a np array"""
  means = np.zeros([d])
  for mvec in title2MVec.values():
    means += mvec.vec
  means /= float(num_movies)
  return means


def get_stds(d, title2MVec, num_movies, means):
  """title2MVec is a dictionary from titles to MovieVecs.
  d is the dimension of all the MovieVecs.
  Return a np array"""
  stds = np.zeros([d])
  for mvec in title2MVec.values():
    stds += (mvec.vec - means)**2
  stds /= float(num_movies)
  stds = (stds)**0.5
  return stds


# def normalize_vec(movie_vec, features_to_normalize, means, stds):
#   """Normalizes a single vector according to given means and stds.
#   Inputs:
#       movie_vec: single movie vec
#       features_to_normalize: a dict from feature names to 0 or 1.
#           We normalize features marked with 1.
#       means and stds: the means and stds over all movie vectors and dimensions
#   Returns:
#       normalized_vec: movie_vec normalized"""
#   d = len(movie_vec)
#   normalized_vec = [0.0] * d
#   for idx,f in enumerate(numerical_features): # non category features
#     if features_to_normalize[f] == 1:
#       normalized_vec[idx] = normalize(movie_vec[idx],means[idx],stds[idx])
#     else:
#       normalized_vec[idx] = movie_vec[idx]
#   start_idx = len(numerical_features) # category features
#   for f in category_features:
#     length = cat_feat_counts[f]
#     for idx in range(length):
#       if features_to_normalize[f] == 1:
#         normalized_vec[start_idx + idx] = normalize(movie_vec[start_idx+idx],means[start_idx+idx],stds[start_idx+idx])
#       else:
#         normalized_vec[start_idx + idx] = movie_vec[start_idx+idx]
#     start_idx = start_idx + length
#   return normalized_vec


def get_normed_vecs(title2MVec):
  """Normalizes all the MovieVecs in the dictionary title2MVec"""
  # Calculate the means and stds
  _, any_MovieVec = title2MVec.popitem() # get any movie...
  d = any_MovieVec.dim() # ...to obtain the length
  num_movies = len(title2MVec.keys()) # get the total number of movies
  means = get_means(d, title2MVec, num_movies)
  stds = get_stds(d, title2MVec, num_movies, means)

  # Normalize the movie vectors
  for _,mvec in title2MVec.iteritems():
    mvec.normalize(means, stds)
  return title2MVec


##########################################################################
# STEP 5: CODE TO CALCULATE DISTANCES BETWEEN MOVIE VECTORS
##########################################################################

def eucl_dist(mvec1,mvec2):
  """Returns euclidean distance between mvec1 and mvec2, which are both MovieVectors"""
  return np.linalg.norm(mvec1.vec-mvec2.vec)

def manh_dist(vec1, vec2):
  """ vec1 and vec2 are both lists of numbers, with the same length.
  This function returns the Manhattan distance between vec1 and vec2."""
  return np.linalg.norm(mvec1.vec-mvec2.vec, 1)



def get_dists(movie_key, feature_weights, distance_function):
    """Calculates the distance from all movies in the database to movie_key,
    using the weights given in feature_weights.
    Returns a list of (dist,key) tuples that doesn't include the original movie."""

    # reweight all the vectors
    title2MVec_reweighted = {}
    for key,vec in title2MVec_norm.iteritems():
        title2MVec_reweighted[key] = reweight_vec(vec,feature_weights)
    movie_vec = title2MVec_reweighted[movie_key]

    # calculate the distances
    dists = []
    for key,vec in title2MVec_reweighted.iteritems():
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

if __name__=="__main__":
  title2movie = import_movies()
  title2movie = postprocess_movies(title2movie)
  cat_feat_items,cat_feat_counts = collect_feat_items(title2movie,category_features)
  title2MVec = create_vectors(title2movie,cat_feat_items)
  title2MVec_norm = get_normed_vecs(title2MVec)
