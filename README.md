# Instructions

## Check if you have Python
###For Macs: 
- Open Terminal (you can type Terminal in your Spotlight Search and open the listed Application). 
- Type “python --version” (without the quotes) into Terminal.  If the output is something like “Python 2.7.11” with perhaps additional words afterwards, then you are good to go!  
- If you get an error message, you have not installed Python.

###For Windows: 
- Open Command Prompt (you can search for it in the search bar) and type “python --version” (without the quotes) into the resulting Command Prompt window. 
- If the output is something like “Python 2.7.11” with perhaps additional words afterwards, then you are good to go!
- If you get an error message, you have not installed Python.

## Setting up Python
- Go to https://www.continuum.io/downloads and download the appropriate version for Mac or Windows.  
- Make sure you download Python version 2.7.
- Open the installer and follow the instructions.
- Check if you successfully installed Python using the previous section’s instructions!

## Download the code
- Go to github.com/abisee/movie-recommender and click the Download ZIP button on the right side.
- All the files you need will now be in a folder called movie-recommender-master.zip which is in your Downloads folder.
- You will need to unzip the file.
- For Mac users, go to the Downloads folder and double click on the movie-recommender-master.zip file
- For PC users, go to the Downloads folder and right click on the movie-recommender-master.zip file and click Extract All. - Make sure the destination is your downloads folder. Then click Extract. 

## Editing the code	
### For Mac users
- Open up your Downloads folder in Finder
- Open the movie-recommender-master folder
- Right click on features.py and click Open With, and select TextEdit
- - Note: if you use TextEdit, you will need to turn off smart quotes in the settings.
- - However, it would be better to download and use a different text editor like Sublime.
- Do the same for titles.txt

###  For Windows users
- Open up your Downloads folder in the File Explorer.
- Click on the movie-recommender-master folder.
- Right click on features.py and click Open With, and select WordPad.
- - Note: you can use WordPad, but it would be better to download and use a different text editor like Sublime.
- Do the same for titles.txt

Every time you **edit** a file, and want the changes to work when you run the program, you must **save** the file.

## Running the code
### For Mac users
- Open Terminal (you can type Terminal in your Spotlight Search and open the listed Application). 
- Into the terminal, type cd Downloads
- This will take you into your Downloads folder.
- Into the terminal, type cd movie-recommender-master
- This will take you into the folder containing all the code files for the exercise
- To run your edited code, type python features.py
- Make sure you have saved your code first!

### For Windows users
- Open Command Prompt (you can search for it in the search bar)
- Into the Command Prompt, type cd Downloads
- This will take you into your Downloads folder.
- Into the terminal, type cd movie-recommender-master
- This will take you into the folder containing all the code files for the exercise
- To run your edited code, type python features.py
- Make sure you have saved your code first!

## What to do
- Try running the code! When you run python features.py in Terminal (Mac) or Command Prompt (Windows), you should see recommendations for the film "Harry Potter and the Goblet of Fire".
- Look at the features.py file in either TextEdit or Notepad and read through it.
- Edit the features (e.g. turn some new features on), and save the file.
- Run the code again. (Tip: instead of typing python features.py repeatedly, you can press the "up" arrow key then press enter.)
- See what your new results are.
- Keep playing around with it until you are happy with the output!
- If you prefer, type a new movie instead of Harry Potter. Make sure you write the title exactly as it appears in titles.txt.

## Things to try
- There are a few foreign movies in the database. Try typing in "Spirited Away" (Japanese) or "Pan's Labyrinth" (Spanish). Do you get other films of the same language or country? Why or why not? How would you adjust the feature weights to get these recommendations?
- There are several movie series in titles.txt, such as the Harry Potter series, Twilight series, Star Wars series and Saw series. When you type in one of these movies, does it output any of the others in the franchise? Why or why not? Can you adjust the feature weights to get these recommendations?
- There are a few famously bad movies in the database. Try typing in "Epic Movie". Do you get other bad movies? Why or why not? Should a good system recommend bad movies?
- After you've adjusted the feature results to give you good recommendations for a particular movie, try entering a completely different movie. Do you still get good recommendations? Your aim is to find feature weights that give you good results for all movies!
- You can turn each feature on or off, i.e. give it weight 1 or 0. But you can actually give each feature any weight you like, like 0.5 (half as important), 2 (twice as important), or 10 (ten times as important), depending on how important you think each feature is. Just no negative numbers!

## If you know Python... (bonus exercises)
- You can try the coding exercise in the recommend.py file. Scroll down to the bottom and look at the manh_dist(vec1, vec2) function. Try filling it in to calculate Manhattan distance, as discussed in the lecture.
- If you're interested, look through the file recommend.py, which contains the code for converting the data contained in movies.txt to the vectors we described in the lecture. See if you can understand what's happening in the code.

