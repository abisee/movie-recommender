from imdb import IMDb

# movies = [
# "the fault in our stars",
# "mean girls",
# "the perks of being a wallflower",
# "wild child",
# "my sister's keeper",
# "clueless",
# "legally blonde",
# "the sisterhood of the traveling pants",
# "when in rome",
# "warm bodies",
# "a cinderella story",
# "twilight",
# "step up 2: the streets",
# # "the princess diaries", # no mpaa
# "so undercover",
# "jack the giant slayer",
# "avatar",
# "a walk to remember",
# "ella enchanted",
# "the fast and the furious",
# "easy A",
# # "the secret life of bees", # nonstandard runtime
# "prom",
# "juno",
# "inception",
# "speak",
# "the hole",
# "x-men origins: wolverine",
# "the 5th wave",
# "tomorrowland",
# "grease",
# "american pie",
# "dazed and confused",
# "x-men",
# "spider-man",
# "21 jump street",
# # "the breakfast club", # no mpaa
# "wild things",
# "boyhood",
# # "karate kid", # no mpaa
# "jumper",
# # "Percy Jackson & the Olympians: The Lightning Thief", # non standard language
# "project almanac",
# "remember the titans",
# "hairspray",
# "carrie",
# "17 again",
# "the place beyond the pines",
# # "footloose", # no mpaa
# "avengers: age of ultron",
# "star wars: the force awakens",
# "minions",
# "inside out",
# "furious 7",
# "the martian",
# "the revenant",
# "ant-man",
# "spectre",
# "the lego movie",
# "transformers: age of extinction",
# "guardians of the galaxy",
# "interstellar",
# # "x-men: days of future past", # nonstandard mpaa
# "the hunger games",
# "the hobbit: the battle of the five armies",
# "maleficent",
# "divergent",
# "edge of tomorrow",
# # "rio 2", # no mpaa
# "how to train your dragon 2",
# "the imitation game",
# "the maze runner",
# "frozen",
# "thor: the dark world",
# "the wolf of wall street",
# "man of steel",
# "iron man 3",
# "the hobbit: the desolation of smaug",
# "the hunger games: catching fire",
# "american hustle",
# "the world's end",
# "gravity",
# "despicable me 2",
# # "monsters university" # no mpaa
# ]

# movies = [
# 'Pulp Fiction',
# 'Schindler\'s List',
# 'The Dark Knight',
# 'Star Wars: Episode V - The Empire Strikes Back',
# 'The Lord of the Rings: The Return of the King',
# 'Star Wars',
# 'Fight Club',
# 'The Lord of the Rings: The Fellowship of the Ring',
# # 'Rear Window',
# # 'Raiders of the Lost Ark',
# # 'Toy Story 3',
# # 'Toy Story 2',
# # 'Psycho',
# 'The Matrix',
# # 'The Silence of the Lambs',
# 'Se7en',
# 'Memento',
# 'It\'s a Wonderful Life',
# 'The Lord of the Rings: The Two Towers',
# 'Forrest Gump',
# # 'Citizen Kane',
# 'Apocalypse Now',
# 'American Beauty',
# # 'Taxi Driver',
# 'Terminator 2: Judgment Day',
# 'Saving Private Ryan',
# # 'Vertigo',
# 'Alien',
# # 'WALL-E',
# # 'The Shining',
# # 'A Clockwork Orange',
# # 'To Kill a Mockingbird',
# 'Aliens',
# 'Eternal Sunshine of the Spotless Mind',
# 'Reservoir Dogs',
# # 'Monty Python and the Holy Grail',
# # 'Back to the Future',
# # 'Singin\' in the Rain',
# # 'Some Like It Hot',
# 'Inglourious Basterds',
# # '2001: A Space Odyssey',
# 'Up',
# 'Gladiator',
# 'Sin City',
# # 'Indiana Jones and the Last Crusade',
# 'Star Wars: Episode VI - Return of the Jedi',
# # 'Die Hard',
# 'Batman Begins',
# # 'Jaws',
# 'Slumdog Millionaire',
# 'Blade Runner',
# # 'Fargo',
# 'No Country for Old Men',
# # 'The Wizard of Oz',
# 'The Sixth Sense',
# 'Donnie Darko',
# 'The Social Network',
# 'Kill Bill: Vol. 1',
# # 'The Lion King',
# 'Million Dollar Baby',
# # 'Toy Story',
# # 'Finding Nemo',
# # 'The Terminator',
# 'How to Train Your Dragon',
# 'V for Vendetta',
# # 'Ratatouille',
# 'The Incredibles',
# 'A Streetcar Named Desire',
# 'The Exorcist',
# 'Kill Bill: Vol. 2',
# 'Kick-Ass',
# 'Pirates of the Caribbean: The Curse of the Black Pearl',
# # 'Monsters, Inc.',
# ]

# movies = [
# # "sharknado",
# "the room",
# "the twilight saga: eclipse",
# "the twilight saga: breaking dawn part 1",
# "the twilight saga: breaking dawn part 2",
# "the twilight saga: new moon",
# "a good day to die hard",
# "saw",
# "saw ii",
# "saw iii",
# "saw iv",
# "saw v",
# "saw vi",
# "disaster movie",
# "epic movie",
# # "the sound of music",
# "casablanca",
# # "dumbo",
# "cinderella",
# # "mary poppins",
# "planet of the apes",
# # "breakfast at tiffany's",
# # "west side story",
# # "the birds",
# # "the rocky horror picture show",
# # "halloween",
# "saturday night fever",
# # "ghostbusters",
# "harry potter sorcerer's stone",
# "harry potter chamber of secrets",
# "harry potter prisoner of azkaban",
# "harry potter goblet of fire",
# "harry potter order of the phoenix",
# "harry potter half blood prince",
# "harry potter deathly hallows part 1",
# "harry potter deathly hallows part 2",
# # "top gun",
# "e.t. the extra-terrestrial",
# # "dirty dancing",
# # "pretty in pink",
# # "sixteen candles",
# "titanic",
# "point break",
# "10 things I hate about you",
# "jurassic park",
# "jurassic world",
# "cruel intentions"
# ]

movies = [
"crouching tiger hidden dragon",
"downfall",
"good bye lenin",
"run lola run",
"the lives of others",
"pan's labyrinth",
"the orphanage",
# "life is beautiful",
"spirited away",
"princess mononoke",
"howl's moving castle",
"godzilla",
# "amelie",
"the pianist",
"persepolis",
# "seven samurai",
# "battle royale",
]

data_out = "more_movies_3.txt"

ia = IMDb()

OUT = open(data_out, 'w')

features_s = ['title', 'mpaa', 'cover url']
features_l = ['genres', 'countries', 'languages', 'runtime']
features_n = ['year', 'rating']

features = features_s + features_l + features_n


def get_mpaa(mpaa_string):
	return mpaa_string.split(" ")[1]


for movie in movies:

	# Get the first search result
	s_result = ia.search_movie(movie)
	result = s_result[0]
	print "Writing out to file: %s" % result['title']
	ia.update(result)

	for f in features:
		OUT.write(f+';')
		if f in features_s:
			if f=='mpaa':
				# print result.keys()
				# print result['certificates']
				OUT.write(get_mpaa(result[f]))
			else:
				OUT.write(result[f])
		elif f in features_l:
			lst = result[f]
			if f=='runtime':
				OUT.write(lst[0])
			else:
				for item in lst:
					OUT.write(item+',')
		elif f in features_n:
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
