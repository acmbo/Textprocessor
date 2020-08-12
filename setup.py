import sys
import subprocess


print('Installing Packages for Textprocessor')

pip install -U spacy pip install -U spacy-lookups-datapython -m spacy download de_core_news_sm
subprocess.check_call([sys.executable, '-m', 'pip', 'install','-U', 'spaCy'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install','-U', 'spaCy-lookups-datapython'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install','download', 'de_core_news_md'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install','-U', 'nltk'])



print('Done')