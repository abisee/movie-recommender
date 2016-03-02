from imdb import IMDb

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
"the princess diaries",
"so undercover",
"jack the giant slayer",
"avatar",
"a walk to remember",
"ella enchanted",
"the fast and the furious",
"easy A",
"the secret life of bees",
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
"the breakfast club",
"wild things",
"boyhood",
"karate kid",
"jumper",
"Percy Jackson & the Olympians: The Lightning Thief",
"project almanac",
"remember the titans",
"hairspray",
"carrie",
"17 again",
"the place beyond the pines",
"footloose",
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
"x-men: days of future past",
"the hunger games",
"the hobbit: the battle of the five armies",
"maleficent",
"divergent",
"edge of tomorrow",
"rio 2",
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
"monsters university"
]

data_out = "movies.txt"

ia = IMDb()

OUT = open(data_out, 'w')
OUT.write("title;year;genres;countries;mpaa;rating;runtime;cover\n")

for movie in movies:
	s_result = ia.search_movie(movie)
	result = s_result[0]
	print "Writing out to file: %s" % result['title']
	ia.update(result)
	# Write out title and year
	OUT.write(result['title'] + ";" + str(result['year']) + ";")
	# Write out genres of the movie
	if 'genre' in result.keys():
		for i in xrange(len(result['genre'])):
			if i == 0:
				OUT.write(result['genre'][i])
				continue
			OUT.write("|" + result['genre'][i])
	OUT.write(";")
	# Write out countries of the movie
	for i in xrange(len(result['countries'])):
		if i == 0:
			OUT.write(result['countries'][i])
			continue
		OUT.write("|" + result['countries'][i])
	# Write out mpaa rating
	if 'mpaa' in result.keys():
		OUT.write(";" + result['mpaa'])
	else:
		OUT.write(";" + "")
   	# Write user rating
   	OUT.write(";" + str(result['rating']))
   	# Write out runtimes
   	OUT.write(";" + str(result['runtimes'][0]))
   	# Write out link to movie poster
   	OUT.write(";" + result['cover url'])
   	OUT.write('\n')

OUT.close()

