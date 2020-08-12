import sys
import subprocess
import time

print('Installing Packages for Textprocessor')
print('-------------------------------------')
print('This Skript will install all dependencys for the Textprocessing Tool. \n The Packages included are: \n -spacy with language Model \n -nltk with stopwords \n -germalemma')

time.sleep(5)
print('-------------------------------------')
#subprocess.check_call([sys.executable, '-m','pip','install','-U', 'spacy'])
#subprocess.call(['python', '-m', 'spacy','download', 'de_core_news_md'])
#subprocess.check_call([sys.executable, '-m','pip','install','-U', 'germanlemma'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install','-U', 'nltk'])
subprocess.call(['python','-m','nltk.downloader' ,'all'])

time.sleep(5)

print('Done')