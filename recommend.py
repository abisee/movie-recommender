from IPython.core.display import HTML
import json
import numpy as np
import copy

data_file = "data/data.json"

# Numerical features in the order they appear in the vectors
numerical_features = ['year', 'rating', 'runtime', 'mpaa', 'votes']

# Category features (i.e. one dimension per genre/country/actor), in the order they appear in the vectors
category_features = ['genres', 'countries', 'languages', 'aspect ratio', 'director', 'cast', 'production companies', 'cinematographer', 'original music', 'producer', 'writer', 'keywords']

# category features to NOT normalize
dont_normalize = ['director', 'cast', 'cinematographer', 'original music', 'producer', 'writer', 'keywords', 'production companies']

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
# Convert runtimes from lists to floats
# Convert dictionary features to individual features
# Take top 3 of all lists
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
    for f,v in movie.iteritems():
      if type(v)==list:
        movie[f] = v[:3]
    if 'mpaa' in movie.keys():
      movie['mpaa'] = mpaa_to_num(movie['mpaa'])
    if 'runtime' in movie.keys():
      runtime_str = movie['runtime'][0]
      try:
        runtime = float(runtime_str)
      except e:
        print "Unusual runtime format: ", runtime_str
        exit()
      movie['runtime'] = runtime
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
            print "unusual parents guidance rating:"
            print fst_sent
            print rating
            print ""
            continue
          movie[f] = rating


##########################################################################
# STEP 2: COLLECT LISTS OF ALL THE GENRES, COUNTRIES, LANGUAGES, ASPECT RATIOS,
# DIRECTORS, ACTORS, PRODUCTION COMPANIES
##########################################################################


def collect_feat_items(title2movie):
  """Collects lists of all category feature values and the number of possible values.
  Inputs:
    title2movie: dict from title to movie feature dict
    category_features: list of features to be collected
  Returns:
    feat2items: dict from feature to list of possible values"""
  feat2items = {f: [] for f in category_features}

  for _,movie in title2movie.iteritems():
    for f in category_features:
      if f not in movie.keys(): continue
      if type(movie[f])==list:
        for item in movie[f]:
          if item not in feat2items[f]:
            feat2items[f].append(item)
      else: # single
        item = movie[f]
        if item not in feat2items[f]:
          feat2items[f].append(item)

  # for f, values in feat2items.iteritems():
  #   print f, len(values)
  #   raw_input("press enter to see next...")
  #   for v in sorted(values):
  #     print v
  #   print ""
  #   raw_input("press enter to see next...")

  return feat2items


##########################################################################
# STEP 3: CREATE A VECTOR FOR EACH MOVIE.
# Each vector has the format: <all the numerical features>, [demographic_features], [parents_guide_features], <all the categorical features>
# Categorical features have one dimension per e.g. actor, with a 0 or 1 value
##########################################################################

def set_vec_layout(feat2items):
  """Determines the length and layout of the movie vectors."""
  feat2indices = {} # maps feature name to list of indices (in the vector)
  item2index = {} # maps item name to single index
  index2feat = {} # maps index to (feature, item) where item may be None
  idx = 0
  for f in numerical_features + demographic_features + parents_guide_features:
    feat2indices[f] = [idx]
    index2feat[idx] = (f, None)
    idx += 1
  for f in category_features:
    num_items = len(feat2items[f])
    feat2indices[f] = range(idx, idx+num_items)
    for item in feat2items[f]:
      item2index[item] = idx
      index2feat[idx] = (f, item)
      idx += 1
  dim = idx
  return dim, feat2indices, item2index, index2feat


class MovieVec:
  """Vector object to hold movie information"""

  def __init__(self, movie, dim, index2feat, feat2indices, item2index):
    """Inputs:
        movie: dictionary
        feat2items: dictionary mapping categorical features to list of all possible items e.g. list of all directors
        dim: dimension of the movie vector
        index2feat: dictionary from index (feature, item) where item may be None
      Note that if features are absent, they are 0 by default"""
    vec = np.zeros([dim])

    for f in numerical_features + demographic_features + parents_guide_features:
      if f in movie.keys():
        idx = feat2indices[f][0]
        vec[idx] = movie[f]

    for f in category_features:
      if f in movie.keys():
        if type(movie[f])==list:
          for item in movie[f]:
            vec[item2index[item]]=1.0
        else:
          vec[item2index[movie[f]]]=1.0


    self.vec = np.array(vec)
    self.dim = dim
    self.index2feat = index2feat
    self.feat2indices = feat2indices

  def normalize(self, means, stds):
    """Subtracts the mean then divides by the std elementwise.
    TODO: change this so that it doesn't normalize some dimensions"""
    assert means.shape == self.vec.shape
    assert stds.shape == self.vec.shape

    for idx in range(self.dim):
      (f, item) = self.index2feat[idx]
      if f not in dont_normalize:
        if stds[idx] == 0.0:
          print "stds for index %i is zero" % idx
          print f, item
        self.vec[idx] = (self.vec[idx] - means[idx]) / stds[idx]

    assert not np.any(np.isinf(self.vec))
    assert not np.any(np.isnan(self.vec))


  def reweight(self, feat2weight):
    """Reweights according to feat2weight, which is a dictionary from features to weights.
    In particular feat2weight has keys numerical_features + demographic_features + parents_guide_features + categorical_features, which should be the same keyset as self.feat2indices."""
    d = self.dim
    for f, wt in feat2weight.iteritems():
      idx_span = self.feat2indices[f]
      for idx in idx_span:
        self.vec[idx] = self.vec[idx] * wt


def create_vectors(title2movie, feat2items):
  """Converts the dictionary representation to a vector representation.
  Inputs:
    title2movie: dict from title to movie feature dict
    feat2items: dict from category feature to list of possible values
  Returns:
    title2MVec: dict from titles to vectors (lists)"""

  print "setting vector layout..."
  dim, feat2indices, item2index, index2feat = set_vec_layout(feat2items)
  title2MVec = {} # will contain vectors
  print "creating MovieVectors..."
  for key,movie in title2movie.iteritems():
    title2MVec[key] = MovieVec(movie, dim, index2feat, feat2indices, item2index)
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


def get_normed_vecs(title2MVec):
  """Normalizes all the MovieVecs in the dictionary title2MVec"""
  # Calculate the means and stds
  _, any_MovieVec = title2MVec.popitem() # get any movie...
  d = any_MovieVec.dim # ...to obtain the length
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
  """Returns Euclidean distance between mvec1 and mvec2, which are both MovieVectors"""
  return np.linalg.norm(mvec1.vec-mvec2.vec)

def manh_dist(mvec1, mvec2):
  """Returns Manhattan distance between mvec1 and mvec2, which are both MovieVectors"""
  return np.linalg.norm(mvec1.vec-mvec2.vec, 1)

def get_dists(query_title, feat2weight, distance_function, title2MVec_norm):
  """Calculates the distance from all movies in the database to query_title,
  using the weights given in feat2weight.
  Returns a list of (dist,key) tuples that doesn't include the original movie."""

  # Copy the movie vecs then reweight
  title2MVec_reweighted = copy.deepcopy(title2MVec_norm)
  for mvec in title2MVec_reweighted.values():
    mvec.reweight(feat2weight)

  query_mvec = title2MVec_reweighted[query_title]

  # Calculate the distances
  dists = [] # will hold ordered list of (title, distance)
  for title,mvec in title2MVec_reweighted.iteritems():
    if title!=query_title:
      dists.append((title, distance_function(query_mvec, mvec)))
  dists_sorted = sorted(dists, key=lambda x:x[1]) # sort by second
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
  """Puts heading n tags around string s"""
  return "<h%i>%s</h%i>" % (n,s,n)

def table_data(s):
  """Puts <td> around input"""
  return "<td>%s</td>" % (s)

def table_header(s):
  """Puts <th> around input"""
  return "<th>%s</th>" % (s)

def table_row(data, header=False):
  """Data is a list of strings.
  Returns a HTML string for a table row."""
  to_return = "<tr>"
  for d in data:
    to_return += table_header(d) if header else table_data(d)
  to_return += "</tr>"
  return to_return

def colgroup(feat2weight,feature_order):
  """Returns a HTML string giving the <col> attributes for the table"""
  to_return = "<colgroup>"
  for f in feature_order:
    if f in ['title', 'distance']:
      to_return += "<col span=\"1\" style=\"background-color:white\">"
    elif feat2weight[f]!=0:
      to_return += "<col span=\"1\" style=\"background-color:white\">" # used
    else:
      to_return += "<col span=\"1\" style=\"background-color:grey\">" # unused
  to_return += "</colgroup>"
  return to_return

def feat_to_info(movie, f, dist):
  """How to format the feature information in the table.
  Returns a string."""
  if f == "distance":
    return "%.3f" % (dist)
  elif f not in movie.keys():
    return "N/A"
  elif type(movie[f])==list:
    return comma_list(movie[f])
  elif type(movie[f])==int:
    return str(movie[f])
  elif type(movie[f])==float:
    if f=="rating":
      return "%.1f" % (movie[f])
    else:
      return "%.3f" % (movie[f])
  else:
    return movie[f]

def feat_to_header(f):
  """How to format the feature name in the header."""
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

def table_row_print(query_title, dist, feature_order, title2movie):
  """Returns a HTML string giving information about the movie
  as a row of a table. Adds dist (which is a float) to the row."""
  movie = title2movie[query_title]
  data = [feat_to_info(movie, f, dist) for f in feature_order]
  return table_row(data)

def get_feature_order(feat2weight):
  """Given the feature weights, determine the order of the columns in the table"""
  feat_wt_tuple_list = sorted(feat2weight.iteritems(), key=lambda x:x[1],reverse=True) # sort by weight
  return ['distance', 'title'] + [f for (f,_) in feat_wt_tuple_list]


def title_match(title1, title2):
  """Determine if two titles match; case and punctuation insensitive. Keeps spaces."""
  punc_set = set(string.punctuation)
  title1 = title1.lower()
  title1 = ''.join(ch for ch in title1 if ch not in punc_set)
  title2 = title2.lower()
  title2 = ''.join(ch for ch in title2 if ch not in punc_set)
  return title1==title2

def search_titles(query_title, title2MVec):
  """Returns a matching title that is in the keys of title2MVec_reweighted. If there is non, returns None"""
  if query_title in title2MVec: return query_title
  for title in title2MVec.keys():
    if title_match(query_title, title): return title
  print "That film isn\'t in the database. Did you type it correctly?"
  return None


def get_recommendations(query_title, feat2weight, title2MVec_norm, title2movie, distance_function=eucl_dist):
  """HTML-prints a table of all movies with given feature weights and distance_function. Returns a HTML string."""
  query_title = search_titles(query_title, title2MVec_norm)
  if query_title is None: return

  feature_order = get_feature_order(feat2weight)
  dists_sorted = get_dists(query_title, feat2weight, distance_function, title2MVec_norm)

  html_string = "<table>"
  html_string += colgroup(feat2weight,feature_order)
  headers = [feat_to_header(f) for f in feature_order]
  html_string += table_row(headers, header=True)
  html_string += table_row_print(query_title, 0, feature_order, title2movie)
  for (key,dist) in dists_sorted:
    html_string += table_row_print(key, dist, feature_order, title2movie)
  html_string += "</table>"
  return HTML(html_string)

##########################################################################
# MAIN. RUNS WHEN IPYTHON NOTEBOOK STARTS
##########################################################################

def init():
  print "importing movies..."
  title2movie = import_movies()
  print "postprocessing..."
  postprocess_movies(title2movie)
  print "collecting items..."
  feat2items = collect_feat_items(title2movie)
  print "creating vectors..."
  title2MVec = create_vectors(title2movie, feat2items)
  print "normalizing..."
  title2MVec_norm = get_normed_vecs(title2MVec)
  return title2MVec_norm, title2movie

# if __name__=="__main__":
  # feat2weight = {}
  # for f in numerical_features + demographic_features + parents_guide_features + category_features:
  #   feat2weight[f] = 1.0

  # title2movie, title2MVec_norm = init() # this is globally visible

  # print get_recommendations("Titanic", feat2weight)
