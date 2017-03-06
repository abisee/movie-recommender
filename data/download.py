"""Download all the titles listed in to_download.txt and save to a .json file"""

import logging
import imdb
import json
import os
import string
import collections

titles_file = "to_download.txt"
json_file = "data.json"

def init_json():
  """Initializes the json file. If it already exists, does nothing"""
  if not os.path.exists(json_file):
    with open(json_file, 'w') as f:
      json.dump([], f)

def read_json():
  """pretty-prints all the info in the json file"""
  with open(json_file, "r") as f:
    data = json.load(f)
  key_counts = collections.Counter()

  for movie in data:
    key_counts.update(movie.keys())

  print "%i movies in total" % len(data)
  for movie in data:
    print movie['runtime']
    print ""


def remove_dups_json():
  """Remove all duplicate movies from json file"""
  with open(json_file, "r") as f:
    data = json.load(f)
  ids = [movie['id'] for movie in data]
  new_data = []
  for idx,movie in enumerate(data):
    if movie['id'] not in ids[:idx]:
      new_data.append(movie)
    else:
      print "duplicate of %s found" % movie['title']
  with open(json_file, "w") as f:
    json.dump(new_data, f)


# def fix_names_json():
#   def remove_leading_space(s):
#     if s=="": return s
#     if s[0]==" ":
#       s = s[1:]
#     return s
#
#   with open(json_file, "r") as f:
#     data = json.load(f)
#   for movie in data:
#     for f,v in movie.iteritems():
#       if isinstance(v, basestring):
#         movie[f] = remove_leading_space(v)
#       elif type(v)==list:
#         for idx,s in enumerate(v):
#           if isinstance(s, basestring):
#             v[idx] = remove_leading_space(s)
#         movie[f] = v
#   with open(json_file, "w") as f:
#     json.dump(data, f)


def append_json(movie_dict):
  """Appends one movie to the json file"""
  if is_in_json_id(movie_dict['id']): return
  with open(json_file, "r") as f:
    feeds = json.load(f)
  with open(json_file, "w") as f:
    feeds.append(movie_dict)
    json.dump(feeds, f)
  print "appended %s to json file" % movie_dict['title']


def title_match(title1, title2):
  """Determine if two titles match; case and punctuation insensitive. Keeps spaces."""
  punc_set = set(string.punctuation)
  title1 = title1.lower()
  title1 = ''.join(ch for ch in title1 if ch not in punc_set)
  title2 = title2.lower()
  title2 = ''.join(ch for ch in title2 if ch not in punc_set)
  return title1==title2


def is_in_json_title(title):
  """Determine if the title is already in the json file. Case and punctuation insensitive."""
  with open(json_file, "r") as f:
    data = json.load(f)
  for movie in data:
    if title_match(title,movie['title']):
      print "%s is already in json file" % title
      return True
  return False

def is_in_json_id(id):
  """Determine if the id is already in the json file"""
  with open(json_file, "r") as f:
    data = json.load(f)
  for movie in data:
    if movie['id']==id:
      print "%s is already in json file" % title
      return True
  return False


def switch_name(name):
  """Given e.g. Potter, Harry return Harry Potter. If no comma found do nothing"""
  parts = name.split(",")
  parts = [s.strip() for s in parts]
  if len(parts)==2:
    return "%s %s" % (parts[1], parts[0])
  else:
    return name

def get_dl_list():
  download_list = []
  with open(titles_file, "r") as f:
    for line in f:
      title = line.strip()
      download_list.append(title)
  return download_list


def convert(thing):
  """Recursively converts thing to a json-writeable format"""
  if isinstance(thing, imdb.Person.Person):
    return switch_name(thing.data['name'])
  elif isinstance(thing, imdb.Company.Company):
    return thing.data['name']
  elif isinstance(thing, imdb.Movie.Movie):
    return (thing['title'], thing.movieID)
  elif type(thing) in [list, tuple]:
    return [convert(x) for x in thing]
  elif type(thing)==dict:
    new_dict = {}
    for key,val in thing.iteritems():
      new_dict[convert(key)] = convert(val)
    return new_dict
  elif type(thing) in [int,float,str]:
    return thing
  elif isinstance(thing, basestring):
    return thing
  elif thing is None:
    return None
  else:
    raise ValueError("did not convert thing: %s" % thing)

def result_to_jsondict(result):
  """Convert search result to json-writeable dictionary"""
  movie_dict = {}
  for k in sorted(result.keys()):
    movie_dict[k] = convert(result[k])

  assert "title" in movie_dict.keys()
  assert 'id' not in movie_dict
  movie_dict['id'] = result.movieID
  return movie_dict


def download_movie(title):
  """Given title which is a string, downloads the info from IMDB and saves to json file. If movie is already in json file, do nothing."""
  if is_in_json_title(title): return
  print "Searching for %s..." % title
  logger = logging.getLogger('imdb')
  logger.setLevel(level=logging.ERROR)
  search_results = ia.search_movie(title)
  r = search_results[0] # take first result
  ia.update(r, 'all') # this fetches all the information
  movie_dict = result_to_jsondict(r)
  append_json(movie_dict)


if __name__=="__main__":
  # ia = imdb.IMDb()
  # init_json()
  # download_list = get_dl_list()
  #
  # for idx,title in enumerate(download_list):
  #   print ""
  #   print "%s of %s; %.2f percent done" % (idx, len(download_list), float(idx)*100/float(len(download_list)))
  #   download_movie(title)

  read_json()
  # remove_dups_json()
  # fix_names_json()
