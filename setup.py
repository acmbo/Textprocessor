import setuptools
from setuptools.command.install import install as _install




class Install(_install):
    def run(self):
        _install.do_egg_install(self)
        import nltk
        nltk.download("stopwords")
        
        import spacy
        print('Downloading language model for the spaCy POS tagger\n')
        from spacy.cli import download
        download('de_core_news_lg')
        #nlp = spacy.load('de_core_news_m')

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Textprocessor-pkg", # Replace with your own username
    version="0.1",
    author="Stephan Wegewitz",
    author_email="sWegewitz@outlook.de",
    description="Package for Extracting Keywords and construction Relationnetworks with them.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    cmdclass={'install': Install},
    install_requires=["scikit-learn","yellowbrick","pandas","numpy","spacy","nltk","germalemma","gensim","pyviz","fuzzywuzzy","python-Levenshtein"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)





    


'''
setuptools.setup(
    name="Textprocessor-pkg", # Replace with your own username
    version="0.1",
    author="Stephan Wegewitz",
    author_email="sWegewitz@outlook.de",
    description="Package for Extracting Keywords and construction Relationnetworks with them.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    cmdclass={'install': Install},
    install_requires=[
      'nltk'],
    setup_requires=['nltk'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)



import spacy

try:
    nlp = spacy.load('de_core_news_lg')
except OSError:
    print('Downloading language model for the spaCy POS tagger\n')
    from spacy.cli import download
    download('de_core_news_lg')
    nlp = spacy.load('de_core_news_lg')
'''