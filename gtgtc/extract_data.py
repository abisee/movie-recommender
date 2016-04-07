from imdb import IMDb
import unicodedata # to handle non-ascii characters

# MOVIES THAT DON'T HAVE THE RIGHT DATA (mainly missing mpaa):

# 'Monty Python and the Holy Grail',
# 'Back to the Future',
# 'Singin\' in the Rain',
# 'Some Like It Hot',
# 'Vertigo',
# 'Taxi Driver',
# 'WALL-E',
# 'The Shining',
# 'A Clockwork Orange',
# 'To Kill a Mockingbird',
# '2001: A Space Odyssey',
# 'Indiana Jones and the Last Crusade',
# 'Die Hard',
# 'The Lion King',
# 'Fargo',
# 'Jaws',
# 'The Wizard of Oz',
# 'Toy Story',
# 'Finding Nemo',
# 'The Terminator',
# 'Ratatouille',
# 'Rear Window',
# 'Raiders of the Lost Ark',
# 'Toy Story 3',
# 'Toy Story 2',
# 'Psycho',
# "carrie",
# "Percy Jackson & the Olympians: The Lightning Thief", # non standard language
# "karate kid", # no mpaa
# "the breakfast club", # no mpaa
# "the secret life of bees", # nonstandard runtime
# "the princess diaries", # no mpaa
# "footloose", # no mpaa
# "x-men: days of future past", # nonstandard mpaa
# "rio 2", # no mpaa
# 'The Silence of the Lambs',
# "monsters university" # no mpaa
# 'Monsters, Inc.',
# "sharknado",
# "top gun",
# "dirty dancing",
# "pretty in pink",
# "sixteen candles",
# "dumbo",
# "the sound of music",
# "mary poppins",
# "breakfast at tiffany's",
# "west side story",
# "the birds",
# "the rocky horror picture show",
# "halloween",
# "ghostbusters",
# "amelie",
# "life is beautiful",

movies = [
"the fault in our stars",
"mean girls",
"the perks of being a wallflower",
"wild child",
"my sister's keeper",
"clueless",
"legally blonde",
"the sisterhood of the traveling pants",
"when in rome",
"warm bodies",
"a cinderella story",
"twilight",
"step up 2: the streets",
"so undercover",
"jack the giant slayer",
"avatar",
"a walk to remember",
"ella enchanted",
"the fast and the furious",
"easy A",
"prom",
"juno",
"inception",
"speak",
"the hole",
"x-men origins: wolverine",
"the 5th wave",
"tomorrowland",
"grease",
"american pie",
"dazed and confused",
"x-men",
"spider-man",
"21 jump street",
"wild things",
"boyhood",
"jumper",
"project almanac",
"remember the titans",
"hairspray",
"17 again",
"the place beyond the pines",
"avengers: age of ultron",
"star wars: the force awakens",
"minions",
"inside out",
"furious 7",
"the martian",
"the revenant",
"ant-man",
"spectre",
"the lego movie",
"transformers: age of extinction",
"guardians of the galaxy",
"interstellar",
"the hunger games",
"the hobbit: the battle of the five armies",
"maleficent",
"divergent",
"edge of tomorrow",
"how to train your dragon 2",
"the imitation game",
"the maze runner",
"frozen",
"thor: the dark world",
"the wolf of wall street",
"man of steel",
"iron man 3",
"the hobbit: the desolation of smaug",
"the hunger games: catching fire",
"american hustle",
"the world's end",
"gravity",
"despicable me 2",
'Pulp Fiction',
'Schindler\'s List',
'The Dark Knight',
'Star Wars: Episode V - The Empire Strikes Back',
'The Lord of the Rings: The Return of the King',
'Star Wars',
'Fight Club',
'The Lord of the Rings: The Fellowship of the Ring',
'The Matrix',
'Se7en',
'Memento',
'It\'s a Wonderful Life',
'The Lord of the Rings: The Two Towers',
'Forrest Gump',
'Apocalypse Now',
'American Beauty',
'Alien',
'Terminator 2: Judgment Day',
'Saving Private Ryan',
'Aliens',
'Eternal Sunshine of the Spotless Mind',
'Reservoir Dogs',
'Inglourious Basterds',
'Up',
'Gladiator',
'Sin City',
'Star Wars: Episode VI - Return of the Jedi',
'Batman Begins',
'Slumdog Millionaire',
'Blade Runner',
'No Country for Old Men',
'The Sixth Sense',
'Donnie Darko',
'The Social Network',
'Kill Bill: Vol. 1',
'Million Dollar Baby',
'How to Train Your Dragon',
'V for Vendetta',
'The Incredibles',
'A Streetcar Named Desire',
'The Exorcist',
'Kill Bill: Vol. 2',
'Kick-Ass',
'Pirates of the Caribbean: The Curse of the Black Pearl',
"the room",
"the twilight saga: eclipse",
"the twilight saga: breaking dawn",
"the twilight saga: new moon",
"a good day to die hard",
"saw",
"saw ii",
"saw iii",
"saw iv",
"saw v",
"saw vi",
"disaster movie",
"epic movie",
"casablanca",
"cinderella",
"planet of the apes",
"saturday night fever",
"harry potter sorcerer's stone",
"harry potter chamber of secrets",
"harry potter prisoner of azkaban",
"harry potter goblet of fire",
"harry potter order of the phoenix",
"harry potter half blood prince",
"harry potter deathly hallows",
"e.t. the extra-terrestrial",
"titanic",
"point break",
"10 things I hate about you",
"jurassic park",
"jurassic world",
"cruel intentions",
"crouching tiger hidden dragon",
"downfall",
"good bye lenin",
"run lola run",
"the lives of others",
"pan's labyrinth",
"the orphanage",
"spirited away",
"princess mononoke",
"howl's moving castle",
"godzilla",
"the pianist",
"persepolis",
]

data_out = "movies.txt"

ia = IMDb()

OUT = open(data_out, 'w')

features_s = ['title', 'mpaa', 'cover url', 'aspect ratio', 'director'] # string features
features_l = ['genres', 'countries', 'languages', 'runtime', 'cast', 'production companies'] # list features
features_n = ['year', 'rating', 'votes'] # numerical features

features = features_s + features_l + features_n

def get_mpaa(mpaa_string):
	"""takes the mpaa_string supplied by imdb and gets the MPAA rating itself"""
	return mpaa_string.split(" ")[1]

def get_person_name(imdb_person):
	"""returns the person's name as a string from an imdb Person object"""
	name = imdb_person.data['name']
	first = name.split(',')[1][1:]
	second = name.split(',')[0]
	return first + ' ' + second

def convert_to_ascii(str):
	"""converts a string to ascii, removing accents etc"""
	return unicodedata.normalize('NFKD',str).encode('ascii','ignore')

# get the features and write them to file
for movie in movies:
	s_result = ia.search_movie(movie)
	result = s_result[0] # Get the first search result
	print "Writing out to file: %s" % result['title']
	ia.update(result) # this is important, not sure why

	for f in features:
		OUT.write(f+';')
		if f in features_s: # string features
			if f=='mpaa':
				OUT.write(get_mpaa(result[f]))
			elif f=='director':
				name = get_person_name(result[f][0])
				name = convert_to_ascii(name)
				OUT.write(name)
			else:
				OUT.write(result[f])
		elif f in features_l: # list features
			lst = result[f]
			if f=='runtime': # just write the first
				OUT.write(lst[0])
			elif f=='production companies': # just write the first
				OUT.write(lst[0]['name'])
			elif f=='cast':
				for item in lst[:3]: # write the first three
					name = get_person_name(item)
					name = convert_to_ascii(name)
					OUT.write(name+',')
			else:
				for item in lst:
					OUT.write(item+',')
		elif f in features_n: # numerical features
			OUT.write(str(result[f]))
		OUT.write('\n')

OUT.close()

# for movie in movies:
# 	# Get the first search result
# 	s_result = ia.search_movie(movie)
# 	if len(s_result)>1:
# 		result = s_result[0]
# 		print "Writing out to file: %s" % result['title']
# 		ia.update(result)
# 		if 'mpaa' in result.keys():
# 			print result['mpaa']
# 		else:
# 			print "no mpaa"
# 	else:
# 		print "no search results"

# feature_counts = {}
#
# for movie in movies:
# 	# Get the first search result
# 	s_result = ia.search_movie(movie)
# 	result = s_result[0]
# 	print result['title']
# 	ia.update(result)
# 	# print result.keys()
# 	for key in result.keys():
# 		# print key
# 		if key in feature_counts.keys():
# 			feature_counts[key] += 1
# 		else:
# 			feature_counts[key] = 1
# 		# print feature_counts
#
# for key,num in feature_counts.iteritems():
# 	print key,num


# for movie in movies:
# 	# Get the first search result
# 	s_result = ia.search_movie(movie)
# 	result = s_result[0]
# 	print result['title']
# 	ia.update(result)
# 	# print result['aspect ratio']
# 	print result['cast']
